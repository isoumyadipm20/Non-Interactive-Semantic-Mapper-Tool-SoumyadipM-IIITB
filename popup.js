document.addEventListener('DOMContentLoaded', function() {
    const sdgGoalsContainer = document.getElementById('sdg-goals-container');
    const sdgTargetsContainer = document.getElementById('sdg-targets-container');
    const sdgIndicatorsContainer = document.getElementById('sdg-indicators-container');
    const colorCodedText = document.getElementById('color-coded-text');
    const slider = document.getElementById('similarity-slider');
    const sliderValue = document.getElementById('slider-value');

    const SERVER_URL = 'http://localhost:5000/process_text';

    const sdgColors = {
        '1': '#e5243b', '2': '#dda63a', '3': '#4c9f38', '4': '#c5192d', '5': '#ff3a21',
        '6': '#26bde2', '7': '#fcc30b', '8': '#a21942', '9': '#fd6925', '10': '#dd1367',
        '11': '#fd9d24', '12': '#bf8b2e', '13': '#3f7e44', '14': '#0a97d9', '15': '#56c02b',
        '16': '#00689d', '17': '#19486a'
    };

    function escapeRegExp(text) {
        return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function getLightColor(hex, opacity = 0.2) {
        let r = parseInt(hex.slice(1, 3), 16),
            g = parseInt(hex.slice(3, 5), 16),
            b = parseInt(hex.slice(5, 7), 16);

        return `rgba(${r}, ${g}, ${b}, ${opacity})`;
    }

    function displayResults(data) {
        const { original_text, goals, targets, indicators, error } = data;

        if (error) {
            // Display the error message
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<p class="error">${error}</p>`;
            resultsDiv.classList.remove('hidden');
            return;
        }

        // Display goals
        const goalsDisplayed = displaySDGItems(goals, sdgGoalsContainer, parseInt(slider.value), 'Goals');

        // Display targets
        const targetsDisplayed = displaySDGItems(targets, sdgTargetsContainer, parseInt(slider.value), 'Targets');

        // Display indicators
        const indicatorsDisplayed = displaySDGItems(indicators, sdgIndicatorsContainer, parseInt(slider.value), 'Indicators');

        // Show error message if no relevant items are found
        if (!goalsDisplayed && !targetsDisplayed && !indicatorsDisplayed) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<p class="error">No relevant items found that meet the threshold.</p>`;
            resultsDiv.classList.remove('hidden');
            return;
        }

        // Add the heading before the color-coded content
        colorCodedText.innerHTML = "<h2>Semantically Mapped Content color-coded to particular SDGs</h2>";

        // Color-code and display text
        let colorCodedHtml = original_text;
        goals.forEach((goal) => {
            const sdgNumber = goal.Id.split('.')[0];
            const color = sdgColors[sdgNumber] || '#000000';
            const lightColor = getLightColor(color);
            goal.relevant_sentences.forEach(sentence => {
                const escapedSentence = escapeRegExp(sentence);
                const regex = new RegExp(`(${escapedSentence})`, 'gi');
                colorCodedHtml = colorCodedHtml.replace(regex, `<span style="background-color: ${lightColor};" title="SDG ${sdgNumber}">$1</span>`);
            });
        });
        colorCodedText.innerHTML += colorCodedHtml;

        const resultsDiv = document.getElementById('results');
        resultsDiv.classList.remove('hidden');
    }

    function displaySDGItems(items, containerElement, threshold, itemType) {
        containerElement.innerHTML = '';
        let itemsDisplayed = false;
        items.filter(item => item.mapping_score >= threshold).forEach((item) => {
            itemsDisplayed = true;
            const sdgNumber = item.Id.split('.')[0];
            const color = sdgColors[sdgNumber] || '#000000';
            const lightColor = getLightColor(color);

            const row = document.createElement('div');
            row.className = 'sdg-row';
            row.innerHTML = `
                <div class="sdg-item" style="flex: 1; background-color: ${lightColor};">${item.Id}</div>
                <div class="sdg-item" style="flex: 4; background-color: ${lightColor};">${item.Description}</div>
                <div class="sdg-item" style="flex: 1; background-color: ${lightColor};">${item.mapping_score.toFixed(2)}%</div>
            `;
            containerElement.appendChild(row);
        });
        return itemsDisplayed;
    }

    function updateDisplay(data) {
        const threshold = parseInt(slider.value);
        sliderValue.textContent = `${threshold}%`;
        displayResults(data);
    }

    let extractedData = null;

    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
        chrome.tabs.executeScript(tabs[0].id, { file: 'content.js' }, function() {
            if (chrome.runtime.lastError) {
                console.error(chrome.runtime.lastError.message);
                return;
            }
            chrome.tabs.sendMessage(tabs[0].id, { action: 'extractText' }, function(response) {
                if (chrome.runtime.lastError) {
                    console.error(chrome.runtime.lastError.message);
                    return;
                }
                if (response && response.text) {
                    fetch(SERVER_URL, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text: response.text })
                    })
                    .then(response => response.json())
                    .then(data => {
                        extractedData = data;
                        updateDisplay(data);
                    })
                    .catch(error => console.error('Error:', error));
                }
            });
        });
    });

    slider.addEventListener('input', function() {
        if (extractedData) {
            updateDisplay(extractedData);
        }
    });
});
