document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById('drawingCanvas');
    const context = canvas.getContext('2d');
    let newCanvas = document.getElementById('newCanvas');
    let newContext = newCanvas.getContext('2d');
    const pred = document.getElementById('pred');
    const historyGrid = document.getElementById('historyGrid');

    context.imageSmoothingEnabled = true;
    context.webkitImageSmoothingEnabled = false;
    context.mozImageSmoothingEnabled = false;

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseleave', stopDrawing);

    clearCanvas();

    let drawing = null;

    function clearCanvas() {
        context.fillStyle = 'black';
        context.fillRect(0, 0, canvas.width, canvas.height);
    }

    function downloadCanvas() {
        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = 'drawing.png';
        link.click();
    }

    function startDrawing(event) {
        drawing = true;
        draw(event);
        pred.textContent = "Inputting...";
    }

    function stopDrawing() {
        drawing = false;
        context.beginPath();
    }

    function draw(event) {
        if (!drawing) return;

        context.lineWidth = 40;
        context.lineCap = 'round';
        context.strokeStyle = 'white';
        const rect = canvas.getBoundingClientRect();
        context.lineTo(event.clientX - rect.left, event.clientY - rect.top);
        context.stroke();
        context.beginPath();
        context.moveTo(event.clientX - rect.left, event.clientY - rect.top);
    }

    function sendData() {
        newContext.drawImage(canvas, 0, 0, 28, 28);

        const grayPixels = getGrayScalePixelData(newContext, 28, 28);
        const data = JSON.stringify({ pixelData: grayPixels });

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: data
        })
        .then(response => response.json())
        .then(data => {
            const result = data.prediction;
            const gridItem = document.createElement('div');
            const image = document.createElement('img');
            const prediction = document.createElement('span');

            gridItem.classList.add('grid-item');
            image.src = newCanvas.toDataURL(); 
            prediction.textContent = `Prediction: ${result}`;

            gridItem.appendChild(image);
            gridItem.appendChild(prediction); 

            historyGrid.appendChild(gridItem); 
            pred.innerHTML = `Did you just write '${result}'?`;
        })
        .catch(error => {
            console.error('Error:', error);
        });

        clearCanvas();
    }

    function clearHistory() {
        while (historyGrid.firstChild) {
            historyGrid.removeChild(historyGrid.firstChild);
        }
    }

    document.getElementById('clear').addEventListener('click', clearCanvas);
    document.getElementById('downloadImage').addEventListener('click', downloadCanvas);
    document.getElementById('getPixelData').addEventListener('click', sendData);
    document.getElementById('clearHistory').addEventListener('click', clearHistory);

    function getGrayScalePixelData(context, width, height) {
        const imageData = context.getImageData(0, 0, width, height);
        const pixels = imageData.data;
        let formattedData = [];

        for (let i = 0; i < 28; i++) {
            let row = [];

            for (let j = 0; j < 28; j++) {
                let index = (i * 28 + j) * 4; 
                let grayscaleValue = pixels[index]; 
                row.push([grayscaleValue]); 
            }

            formattedData.push(row);
        }
        formattedData = [formattedData]; 

        return formattedData;
    }
});
