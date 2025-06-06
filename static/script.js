document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const featureVizEl = document.getElementById('feature-viz');
    const featureSlidersEl = document.getElementById('feature-sliders');
    
    let featureProjections = [];
    let activeFeatures = [];
    let featureViz = null;
    let featureSliders = {};
    
    function getFeatureColor(featureId) {
        const hue = (featureId * 137.5) % 360;
        return `hsl(${hue}, 70%, 50%)`;
    }
    
    let socket = null;
    let isGenerating = false;
    
    function initFeatureViz() {
        const width = featureVizEl.clientWidth;
        const height = featureVizEl.clientHeight;
        
        const margin = {top: 20, right: 20, bottom: 20, left: 20};
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        
        d3.select(featureVizEl).selectAll('svg').remove();
        
        featureViz = d3.select(featureVizEl)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${margin.left}, ${margin.top})`);
            
        featureViz.append('rect')
            .attr('width', innerWidth)
            .attr('height', innerHeight)
            .attr('fill', '#f9f9f9')
            .attr('stroke', '#eaeaea')
            .attr('rx', 4)
            .attr('ry', 4);
        
        const gridSize = 4;
        for (let i = 1; i < gridSize; i++) {
            featureViz.append('line')
                .attr('x1', (innerWidth / gridSize) * i)
                .attr('y1', 0)
                .attr('x2', (innerWidth / gridSize) * i)
                .attr('y2', innerHeight)
                .attr('stroke', '#eaeaea')
                .attr('stroke-dasharray', '2,2');
                
            featureViz.append('line')
                .attr('x1', 0)
                .attr('y1', (innerHeight / gridSize) * i)
                .attr('x2', innerWidth)
                .attr('y2', (innerHeight / gridSize) * i)
                .attr('stroke', '#eaeaea')
                .attr('stroke-dasharray', '2,2');
        }
    }
    
    function updateFeatureViz() {
        if (!featureViz) return;
        
        if (!featureProjections || featureProjections.length === 0) {
            return;
        }
        
        const width = featureVizEl.clientWidth;
        const height = featureVizEl.clientHeight;
        const margin = {top: 20, right: 20, bottom: 20, left: 20};
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        
        const activeIds = activeFeatures.map(f => f.id);
        
        const visibleFeatureProjections = activeFeatures
            .filter(f => f.value > 0)
            .map(f => {
                const proj = featureProjections.find(p => p.id === f.id);
                if (proj) {
                    return { ...proj, value: f.value };
                }
                return null;
            })
            .filter(Boolean);
            
        let xExtent, yExtent;
        
        if (visibleFeatureProjections.length > 0) {
            xExtent = d3.extent(visibleFeatureProjections, d => d.x);
            yExtent = d3.extent(visibleFeatureProjections, d => d.y);
        } else {
            xExtent = d3.extent(featureProjections, d => d.x);
            yExtent = d3.extent(featureProjections, d => d.y);
        }
        
        const xRange = Math.max(xExtent[1] - xExtent[0], 0.1);
        const yRange = Math.max(yExtent[1] - yExtent[0], 0.1);
        
        const xPadding = xRange * 0.4;
        const yPadding = yRange * 0.4;
        
        const xScale = d3.scaleLinear()
            .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
            .range([0, innerWidth]);
        
        const yScale = d3.scaleLinear()
            .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
            .range([innerHeight, 0]); // Flip y-axis for correct orientation
            
        if (activeFeatures.length === 0 || visibleFeatureProjections.length === 0) {
            featureViz.selectAll('.feature-circle').remove();
            featureViz.selectAll('.feature-label').remove();
            return;
        }
            
        const circles = featureViz.selectAll('.feature-circle')
            .data(visibleFeatureProjections, d => d.id);
            
        circles.exit()
            .transition()
            .duration(300)
            .attr('r', 0)
            .remove();
        
        circles.enter()
            .append('circle')
            .attr('class', 'feature-circle')
            .attr('id', d => `feature-${d.id}`)
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 0)
            .attr('fill', d => getFeatureColor(d.id))
            .attr('stroke', '#fff')
            .attr('stroke-width', 1)
            .attr('cursor', 'pointer')
            .on('click', (event, d) => addFeatureSlider(d.id))
            .merge(circles)
            .transition()
            .duration(300)
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', d => {
                return Math.sqrt(d.value) * 15;
            })
            .attr('fill-opacity', 0.7)
            .attr('stroke-opacity', 0.8);
            
        const labels = featureViz.selectAll('.feature-label')
            .data(visibleFeatureProjections, d => d.id);
            
        labels.exit().remove();
        
        labels.enter()
            .append('text')
            .attr('class', 'feature-label')
            .attr('x', d => xScale(d.x))
            .attr('y', d => yScale(d.y))
            .attr('dy', '0.35em')
            .attr('text-anchor', 'middle')
            .style('fill', '#333')
            .style('font-size', '10px')
            .style('font-weight', '500')
            .style('pointer-events', 'none')
            .style('user-select', 'none')
            .text(d => d.id)
            .merge(labels)
            .transition()
            .duration(300)
            .attr('x', d => xScale(d.x))
            .attr('y', d => yScale(d.y));
    }
    
    function addFeatureSlider(featureId) {
        if (featureSliders[featureId]) return;
        
        const emptyMessage = featureSlidersEl.querySelector('.empty-sliders-message');
        if (emptyMessage) {
            emptyMessage.remove();
        }
        
        const sliderContainer = document.createElement('div');
        sliderContainer.className = 'slider-container';
        sliderContainer.id = `slider-container-${featureId}`;
        
        const labelElement = document.createElement('label');
        labelElement.innerHTML = `
            Feature ${featureId}
            <span class="slider-value">1.0</span>
        `;
        
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = '3';
        slider.step = '0.1';
        slider.value = '1.0';
        
        featureSliders[featureId] = {
            container: sliderContainer,
            input: slider,
            valueDisplay: labelElement.querySelector('.slider-value')
        };
        
        const color = getFeatureColor(featureId);
        slider.style.accentColor = color;
        
        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            featureSliders[featureId].valueDisplay.textContent = value.toFixed(1);
            
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({
                    type: 'scale_feature',
                    id: featureId,
                    value: value
                }));
            }
        });
        
        const removeButton = document.createElement('button');
        removeButton.innerHTML = '&times;';
        removeButton.className = 'remove-slider';
        removeButton.addEventListener('click', () => {
            sliderContainer.remove();
            delete featureSliders[featureId];
            
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({
                    type: 'scale_feature',
                    id: featureId,
                    value: 1.0
                }));
            }
            
            if (Object.keys(featureSliders).length === 0) {
                const emptyMsg = document.createElement('p');
                emptyMsg.className = 'empty-sliders-message';
                emptyMsg.textContent = 'Click on features to add controls';
                featureSlidersEl.appendChild(emptyMsg);
            }
        });
        
        sliderContainer.appendChild(labelElement);
        sliderContainer.appendChild(slider);
        sliderContainer.appendChild(removeButton);
        
        featureSlidersEl.appendChild(sliderContainer);
    }
    
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/generate`;
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log('WebSocket connection established');
        };
        
        socket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            
            if (message.type === 'set_text') {
                textInput.value = message.content;
                textInput.scrollTop = textInput.scrollHeight;
            } else if (message.type === 'token') {
                textInput.value += message.content;
                textInput.scrollTop = textInput.scrollHeight;
            } else if (message.type === 'average_activations') {
                if (message.features.projections && message.features.projections.length > 0) {
                    const idProjections = message.features.activations.map((act, idx) => {
                        if (idx < message.features.projections.length) {
                            const proj = message.features.projections[idx];
                            return {
                                id: act.id,
                                x: proj[0],
                                y: proj[1]
                            };
                        }
                        return null;
                    }).filter(Boolean);
                    
                    idProjections.forEach(newProj => {
                        const existingIdx = featureProjections.findIndex(p => p.id === newProj.id);
                        if (existingIdx === -1) {
                            featureProjections.push(newProj);
                        } else {
                            featureProjections[existingIdx] = newProj;
                        }
                    });
                }
                
                activeFeatures = message.features.activations;
                updateFeatureViz();
                
                const vizTitle = document.querySelector('#feature-viz-container h3');
                if (vizTitle) {
                    vizTitle.textContent = 'Feature Activations';
                    vizTitle.classList.add('averaged');
                }
            } else if (message.type === 'feature_projection') {
                featureProjections = message.projections;
            } else if (message.type === 'error') {
                console.error('Error from server:', message.content);
                stopGeneration();
            } else if (message.type === 'end') {
                stopGeneration();
            }
        };
        
        socket.onclose = () => {
            console.log('WebSocket connection closed');
            socket = null;
            stopGeneration();
        };
        
        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            stopGeneration();
        };
    }
    
    function startGeneration() {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            connectWebSocket();
            setTimeout(() => startGeneration(), 500);
            return;
        }
        
        isGenerating = true;
        playPauseBtn.classList.remove('play');
        playPauseBtn.classList.add('pause');
        
        const vizTitle = document.querySelector('#feature-viz-container h3');
        if (vizTitle) {
            vizTitle.textContent = 'Feature Activations';
            vizTitle.classList.remove('averaged');
        }
        
        activeFeatures = [];
        updateFeatureViz();
        
        socket.send(JSON.stringify({
            type: 'start',
            prompt: textInput.value
        }));
    }
    
    function stopGeneration() {
        isGenerating = false;
        playPauseBtn.classList.remove('pause');
        playPauseBtn.classList.add('play');
        
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: 'stop'
            }));
        }
    }
    
    function toggleGeneration() {
        if (isGenerating) {
            stopGeneration();
        } else {
            startGeneration();
        }
    }
    
    playPauseBtn.addEventListener('click', toggleGeneration);
    
    initFeatureViz();
    
    connectWebSocket();
    
    window.addEventListener('resize', () => {
        initFeatureViz();
        updateFeatureViz();
    });
});
