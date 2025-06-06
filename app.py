import asyncio
import json
import torch
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import umap
import uvicorn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List, Optional, Tuple

from transformer_lens import HookedTransformer, utils
from sae import SAE

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=1)

model = None
tokenizer = None
sae_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
active_generations = {}

SAE_LAYER = 13
TOP_K_FEATURES = 16
SAVE_PATH = "sae.pt"
feature_umap = None
feature_scaling = {}
generation_activations = {}


def process_and_store_activations(
    resid_post: torch.Tensor, connection_id: int, store=True
) -> Tuple[List[Dict], List[Tuple[float, float]]]:
    if sae_model is None or feature_umap is None:
        return [], []

    try:
        with torch.no_grad():
            sae_acts = sae_model.encode(resid_post.unsqueeze(0))
            sae_acts = sae_acts.squeeze(0).cpu().numpy()

        scaled_acts = {}
        for i in range(sae_acts.shape[0]):
            value = float(sae_acts[i])
            if value > 0:
                scale = feature_scaling.get(i, 1.0)
                scaled_acts[i] = value * scale

        if connection_id is not None and store:
            if connection_id not in generation_activations:
                generation_activations[connection_id] = []

            generation_activations[connection_id].append(scaled_acts)

        activations = [{"id": i, "value": v} for i, v in scaled_acts.items()]

        activations.sort(key=lambda x: x["value"], reverse=True)

        top_k = min(TOP_K_FEATURES, len(activations))
        activations = activations[:top_k]

        feature_ids = [a["id"] for a in activations]
        projections = []

        max_features = len(feature_umap.embedding_)
        for feat_id in feature_ids:
            if feat_id < max_features:
                feature_vector = (
                    sae_model.w_d.data[:, feat_id].cpu().numpy().reshape(1, -1)
                )
                coord = feature_umap.transform(feature_vector)[0]
                projections.append((float(coord[0]), float(coord[1])))
            else:
                projections.append((float(np.random.randn()), float(np.random.randn())))

        return activations, projections

    except Exception as e:
        print(f"Error processing activations: {e}")
        import traceback

        traceback.print_exc()
        return [], []


def calculate_average_activations(connection_id) -> List[Dict]:
    try:
        if (
            connection_id not in generation_activations
            or not generation_activations[connection_id]
        ):
            return []

        feature_sums = {}
        feature_counts = {}

        for token_activations in generation_activations[connection_id]:
            for feature_id, value in token_activations.items():
                if feature_id not in feature_sums:
                    feature_sums[feature_id] = 0
                    feature_counts[feature_id] = 0

                feature_sums[feature_id] += value
                feature_counts[feature_id] += 1

        avg_activations = []
        for feature_id, total in feature_sums.items():
            count = feature_counts[feature_id]
            avg_value = total / count if count > 0 else 0

            scale = feature_scaling.get(feature_id, 1.0)
            scaled_value = avg_value * scale

            if scaled_value > 0:
                avg_activations.append(
                    {
                        "id": feature_id,
                        "value": scaled_value,
                        "raw_value": avg_value,
                        "scale": scale,
                        "count": count,
                        "percentage": count
                        / len(generation_activations[connection_id])
                        * 100,
                    }
                )

        avg_activations.sort(key=lambda x: x["value"], reverse=True)

        top_k = min(TOP_K_FEATURES, len(avg_activations))
        avg_activations = avg_activations[:top_k]

        return avg_activations

    except Exception as e:
        print(f"Error calculating average activations: {e}")
        import traceback

        traceback.print_exc()
        return []


async def load_model():
    global model, tokenizer, sae_model, feature_umap

    print("Loading model...")
    model_name = "google/gemma-2-2b"

    try:
        model = HookedTransformer.from_pretrained(model_name, device=device)
        tokenizer = model.tokenizer
        model.compile()
        print(f"Model {model_name} loaded successfully on {device}")

        print("Loading SAE model...")
        d_model = model.cfg.d_model

        sae_params = {
            "d_model": d_model,
            "n_features": 50_000,
            "bandwidth": 2.0,
            "threshold": 0.1,
            "lambda_p": 3e-6,
            "c": 4.0,
        }
        sae_model = SAE(**sae_params)

        try:
            print(f"Loading SAE weights from {SAVE_PATH}...")
            state_dict = torch.load("sae.pt", map_location=device)
            sae_model.load_state_dict(state_dict)
            sae_model.to(device)
            sae_model.eval()

            actual_feature_count = sae_model.w_d.shape[1]
            print(f"Loaded SAE model with {actual_feature_count} features")

            for i in range(actual_feature_count):
                feature_scaling[i] = 1.0

            print("Initializing UMAP model for feature visualization...")
            feature_directions = sae_model.w_d.data.cpu().numpy().T
            max_features_for_umap = min(10000, feature_directions.shape[0])
            print(f"Computing UMAP on {max_features_for_umap} features")
            feature_umap = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            feature_umap.fit(feature_directions[:max_features_for_umap])
        except Exception as e:
            print(f"Error loading SAE weights: {e}")
            import traceback

            traceback.print_exc()
            sae_model = None
            feature_umap = None

        print("SAE model loaded successfully")

    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        tokenizer = None
        sae_model = None


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_model())


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_token(self, websocket: WebSocket, token: str, connection_id: int):
        await websocket.send_json({"type": "token", "content": token})

    async def send_average_activations(self, websocket: WebSocket, connection_id: int):
        if sae_model is None or feature_umap is None:
            return

        avg_activations = calculate_average_activations(connection_id)

        feature_ids = [act["id"] for act in avg_activations]
        projections = []

        feature_directions = sae_model.w_d.data.cpu().numpy().T
        max_features = len(feature_umap.embedding_)

        for feat_id in feature_ids:
            if feat_id < max_features:
                feature_vector = feature_directions[feat_id].reshape(1, -1)
                coord = feature_umap.transform(feature_vector)[0]
                projections.append((float(coord[0]), float(coord[1])))
            else:
                projections.append((float(np.random.randn()), float(np.random.randn())))

        await websocket.send_json(
            {
                "type": "average_activations",
                "features": {
                    "activations": avg_activations,
                    "projections": projections,
                },
            }
        )

    async def send_error(self, websocket: WebSocket, error: str):
        await websocket.send_json({"type": "error", "content": error})

    async def send_feature_projection(self, websocket: WebSocket):
        if sae_model is None or feature_umap is None:
            return

        try:
            n_features = sae_model.w_d.shape[1]

            max_features = min(5000, n_features)
            max_features = min(max_features, len(feature_umap.embedding_))

            batch_size = 500
            all_projections = []

            for start_idx in range(0, max_features, batch_size):
                end_idx = min(start_idx + batch_size, max_features)
                batch_indices = list(range(start_idx, end_idx))

                feature_vectors = sae_model.w_d.data[:, batch_indices].cpu().numpy().T
                coords = feature_umap.transform(feature_vectors)

                batch_projections = [
                    {"id": i, "x": float(coords[j][0]), "y": float(coords[j][1])}
                    for j, i in enumerate(batch_indices)
                ]

                all_projections.extend(batch_projections)

            await websocket.send_json(
                {"type": "feature_projection", "projections": all_projections}
            )

        except Exception as e:
            print(f"Error sending feature projections: {e}")
            import traceback

            traceback.print_exc()


manager = ConnectionManager()


@app.websocket("/ws/generate")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    connection_id = id(websocket)
    active_generations[connection_id] = {"stop_requested": False}

    if connection_id in generation_activations:
        generation_activations[connection_id] = []

    await manager.send_feature_projection(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "start":
                prompt = message.get("prompt", "")

                generation_activations[connection_id] = []

                asyncio.create_task(generate_text(websocket, connection_id, prompt))

            elif message["type"] == "stop":
                active_generations[connection_id]["stop_requested"] = True

                if (
                    connection_id in active_generations
                    and "residual_activations" in active_generations[connection_id]
                ):
                    stored_activations = active_generations[connection_id][
                        "residual_activations"
                    ]

                    if stored_activations and sae_model is not None:
                        if connection_id in generation_activations:
                            generation_activations[connection_id] = []

                        for act in stored_activations:
                            act = act.to(device)
                            process_and_store_activations(
                                act, connection_id, store=True
                            )

                        active_generations[connection_id]["residual_activations"] = []

                    if (
                        connection_id in generation_activations
                        and len(generation_activations[connection_id]) > 1
                    ):
                        await manager.send_average_activations(websocket, connection_id)

            elif message["type"] == "scale_feature":
                feature_id = message.get("id")
                scale_value = message.get("value", 1.0)

                if feature_id is not None and feature_id in feature_scaling:
                    feature_scaling[feature_id] = float(scale_value)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if connection_id in active_generations:
            del active_generations[connection_id]

        if connection_id in generation_activations:
            del generation_activations[connection_id]


async def generate_text(websocket: WebSocket, connection_id: int, prompt: str):
    if model is None or tokenizer is None:
        await manager.send_error(
            websocket, "Model is still loading, please try again in a moment"
        )
        return

    active_generations[connection_id]["stop_requested"] = False

    try:
        await websocket.send_json({"type": "set_text", "content": prompt})

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        last_text = prompt

        for _ in range(100):
            if (
                connection_id not in active_generations
                or active_generations[connection_id]["stop_requested"]
            ):
                break

            loop = asyncio.get_event_loop()

            resid_post_activations = []

            def hook_fn(act, hook):
                resid_post_activations.append(act.detach())

            output_logits = await loop.run_in_executor(
                executor,
                lambda: model.run_with_hooks(
                    input_ids,
                    fwd_hooks=[(utils.get_act_name("resid_post", SAE_LAYER), hook_fn)],
                ),
            )

            next_token_id = torch.argmax(output_logits[:, -1, :], dim=-1).item()

            if next_token_id is None:
                await manager.send_error(websocket, f"Error generating text")
                break

            if resid_post_activations and sae_model is not None:
                last_token_act = resid_post_activations[0][0, -1].to(device)

                if connection_id not in active_generations:
                    active_generations[connection_id] = {
                        "stop_requested": False,
                        "residual_activations": [],
                    }
                if "residual_activations" not in active_generations[connection_id]:
                    active_generations[connection_id]["residual_activations"] = []

                active_generations[connection_id]["residual_activations"].append(
                    last_token_act.detach().cpu()
                )

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=device)], dim=1
            )

            full_text = tokenizer.decode(input_ids[0])

            new_content = full_text[len(last_text) :]
            last_text = full_text

            if new_content:
                await manager.send_token(websocket, new_content, connection_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            await asyncio.sleep(0.05)

    except Exception as e:
        print(f"Error in generate_text: {e}")
        await manager.send_error(websocket, f"Error: {str(e)}")

    await websocket.send_json({"type": "end"})


@app.post("/start")
async def start_generation():
    return {"status": "started"}


@app.post("/stop")
async def stop_generation():
    return {"status": "stopped"}


os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
