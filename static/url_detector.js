// URL Detector JavaScript

document.getElementById('urlForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const url = document.getElementById('url').value.trim();
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    
    // Hide result, show loading
    resultDiv.classList.add('hidden');
    loadingDiv.classList.remove('hidden');
    
    try {
        const response = await fetch('/api/check-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResult(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error checking URL: ' + error.message);
    } finally {
        loadingDiv.classList.add('hidden');
    }
});

function displayResult(data) {
    const resultDiv = document.getElementById('result');
    
    // Set URL
    document.getElementById('resultUrl').textContent = data.url;
    
    // Set prediction
    const predictionBadge = document.getElementById('predictionBadge');
    const predictionText = document.getElementById('predictionText');
    
    if (data.is_phishing) {
        predictionBadge.className = 'prediction-badge phishing';
        predictionText.textContent = '⚠️ PHISHING';
    } else {
        predictionBadge.className = 'prediction-badge legitimate';
        predictionText.textContent = '✓ LEGITIMATE';
    }
    
    // Set confidence
    const confidencePercent = (data.confidence * 100).toFixed(1);
    document.getElementById('confidenceFill').style.width = confidencePercent + '%';
    document.getElementById('confidenceText').textContent = confidencePercent + '%';
    
    // Set risk level
    const riskBadge = document.getElementById('riskLevel');
    riskBadge.textContent = data.risk_level;
    riskBadge.className = 'risk-badge ' + data.risk_level.toLowerCase();
    
    // Show/hide warning
    const warningBox = document.getElementById('warningBox');
    const safeBox = document.getElementById('safeBox');
    
    if (data.is_phishing) {
        warningBox.classList.remove('hidden');
        safeBox.classList.add('hidden');
    } else {
        warningBox.classList.add('hidden');
        safeBox.classList.remove('hidden');
    }
    
    // Show result
    resultDiv.classList.remove('hidden');
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
