// File Version: 1.2.0
// /static/ui_handlers.js

// # Precision File Search
// # Copyright (c) 2025 Ali Kazemi
// # Licensed under MPL 2.0
// # This file is part of a derivative work and must retain this notice.

// 1. IMPORTS ####################################################################################################
import {
    currentTranslations, finishSound, escapeHTML, includeDotFoldersCheckbox,
    aiSearchForm, aiQueryInput, aiSearchButton, aiResultsSection, aiResultsDiv,
    aiTemperatureSlider, aiTemperatureValue, aiMaxTokensSlider, aiMaxTokensValue,
    startIndexingButton, semanticPathInput, semanticSearchSection, semanticSearchForm,
    semanticQueryInput, semanticResultsSection, semanticResultsDiv, enableRerankerToggle,
    rerankerOptionsDiv, quickKFetch, quickVectorScoreThreshold, quickVectorTopN,
    quickScoreThreshold, quickTopN, startClassificationButton, classifierPathInput,
    progressSection, progressBar, statusText, resultsSectionClassifier, resultsAccordion,
    organizeAllButton, enableRerankerSetting, startTrainingButton, trainingDataPathInput,
    testSizeSlider, nEstimatorsInput, trainerStatusSection, trainerLog, trainerResult,
    modalOverlay, modalTitle, modalText, modalInput, modalConfirmBtn, modalCancelBtn,
    modalCloseBtn, modalErrorText
} from './main_app.js';


// 2. MODULE STATE ###############################################################################################
let classificationPoller = null;
let indexingPoller = null;
let trainingPoller = null;
let lastSemanticResults = [];


// 3. AI SEARCH HANDLERS #########################################################################################
export function initializeAISearchUI(defaults) {
    const params = defaults?.ai_search_params || {};
    aiTemperatureSlider.value = localStorage.getItem('ai_temperature') || params.default_temperature || 0.2;
    aiTemperatureValue.textContent = aiTemperatureSlider.value;
    aiMaxTokensSlider.value = localStorage.getItem('ai_max_tokens') || params.default_max_tokens || 4096;
    aiMaxTokensValue.textContent = aiMaxTokensSlider.value;
}

// Block Version: 1.2.0
function renderAIResults(data) {
    aiResultsDiv.innerHTML = '';
    aiResultsSection.removeAttribute('dir'); 

    const currentLangDirection = document.documentElement.dir;
    if (currentLangDirection === 'rtl') {
        aiResultsSection.setAttribute('dir', 'rtl');
    }

    const summaryDiv = document.createElement('div');
    summaryDiv.className = 'markdown-body';

    const rawHtml = marked.parse(data.summary || '<p>No summary provided.</p>');
    const sanitizedHtml = DOMPurify.sanitize(rawHtml);
    summaryDiv.innerHTML = sanitizedHtml;

    aiResultsDiv.appendChild(summaryDiv);

    if (!data.relevant_files || data.relevant_files.length === 0) return;

    const filesHeader = document.createElement('h3');
    filesHeader.className = 'modal-subtitle';
    filesHeader.innerHTML = `<i class="fas fa-file-invoice"></i> ${currentTranslations.keyFilesFound || 'Key Files Found'}`;
    aiResultsDiv.appendChild(filesHeader);

    data.relevant_files.forEach(result => {
        const item = document.createElement('div');
        item.className = 'semantic-result-item';
        let scoresHTML = '';
        if (result.vector_score !== undefined && result.vector_score !== null) {
            scoresHTML += `<span class="score-item" title="Vector Similarity Score"><span class="score-label">Vector:</span><span class="score-value">${result.vector_score.toFixed(4)}</span></span>`;
        }
        if (result.rerank_score !== undefined && result.rerank_score !== null) {
            scoresHTML += `<span class="score-item" title="Rerank Relevance Score"><span class="score-label">Reranker:</span><span class="score-value">${result.rerank_score.toFixed(4)}</span></span>`;
        }
        item.innerHTML = `
            <div class="result-item-header">
                <span class="result-path">${escapeHTML(result.path)}</span>
                <div class="result-actions">
                    ${scoresHTML}
                    <i class="fas fa-folder-open action-btn" data-path="${escapeHTML(result.path)}" data-action="folder" title="Open Containing Folder"></i>
                    <i class="fas fa-file-alt action-btn" data-path="${escapeHTML(result.path)}" data-action="file" title="Open File"></i>
                </div>
            </div>
            <div class="result-chunk"><p>${escapeHTML(result.relevance)}</p></div>
        `;
        aiResultsDiv.appendChild(item);
    });
}


export async function performAISearch(event) {
    event.preventDefault();
    finishSound.play().catch(() => {}); finishSound.pause(); finishSound.currentTime = 0;
    const query = aiQueryInput.value;
    if (!query) return;

    aiSearchButton.disabled = true;
    aiSearchButton.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${currentTranslations.aiSearching || 'ASK AI'}`;
    aiResultsSection.classList.remove('hidden');
    aiResultsDiv.innerHTML = '<div class="spinner-container"><div class="loading-spinner"></div><p>AI is processing your request...</p></div>';

    let endpoint = '/api/ai/search';
    let body = {};

    if (lastSemanticResults && lastSemanticResults.length > 0) {
        console.log("Context found. Using /api/ai/summarize-results endpoint.");
        endpoint = '/api/ai/summarize-results';
        body = {
            query: query,
            search_results: lastSemanticResults,
            temperature: parseFloat(aiTemperatureSlider.value),
            max_tokens: parseInt(aiMaxTokensSlider.value, 10),
        };
    } else {
        console.log("No context found. Using /api/ai/search endpoint.");
        const semanticParams = {
            k_fetch_initial: parseInt(quickKFetch.value, 10),
            vector_score_threshold: parseFloat(quickVectorScoreThreshold.value),
            vector_top_n: parseInt(quickVectorTopN.value, 10),
            enable_reranker: enableRerankerToggle.checked,
            rerank_score_threshold: parseFloat(quickScoreThreshold.value),
            rerank_top_n: parseInt(quickTopN.value, 10)
        };
        endpoint = '/api/ai/search';
        body = {
            query: query,
            temperature: parseFloat(aiTemperatureSlider.value),
            max_tokens: parseInt(aiMaxTokensSlider.value, 10),
            ...semanticParams
        };
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const result = await response.json();
        if (!response.ok) {
            const errorData = result.detail?.data || { summary: result.detail || 'An unknown server error occurred.' };
            renderAIResults(errorData);
            return;
        }
        if (result.status === 'success') {
            renderAIResults(result.data);
            finishSound.play().catch(error => console.error("Audio playback failed:", error));
        } else {
            throw new Error(result.message || 'The search failed to produce a result.');
        }
    } catch (error) {
        renderAIResults({ summary: `<h3>Error</h3><p>${error.message}</p>` });
    } finally {
        aiSearchButton.disabled = false;
        aiSearchButton.innerHTML = `<i class="fas fa-paper-plane"></i> <span>${currentTranslations.aiSearchButton || 'ASK AI'}</span>`;
    }
}


export async function handleAIResultsClick(event) {
    const button = event.target.closest('.action-btn');
    if (!button) return;
    const originalClass = button.className;
    button.className = 'fas fa-spinner fa-spin action-btn';
    try {
        await fetch('/api/open', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: button.dataset.path, action: button.dataset.action })
        });
    } catch (error) {
        console.error('Failed to open path:', error);
        alert(`Failed to open path: ${button.dataset.path}`);
    } finally {
        setTimeout(() => { button.className = originalClass; }, 500);
    }
}

// 4. SEMANTIC SEARCH HANDLERS ###################################################################################
export async function startIndexing() {
    finishSound.play().catch(() => {}); finishSound.pause(); finishSound.currentTime = 0;
    const path = semanticPathInput.value; if (!path) { alert("Please enter a directory path to index."); return; }

    startIndexingButton.disabled = true;
    startIndexingButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> STARTING...';
    try {
        const response = await fetch('/api/semantic/start-indexing', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ search_path: path, include_dot_folders: includeDotFoldersCheckbox.checked })
        });
        if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'Failed to start indexing.'); }
        indexingPoller = setInterval(updateIndexingProgress, 1500);
    } catch (error) {
        startIndexingButton.disabled = false;
        startIndexingButton.innerHTML = `<i class="fas fa-times"></i> FAILED TO START`;
        setTimeout(() => {
             startIndexingButton.innerHTML = `<i class="fas fa-play-circle"></i> <span data-i18n-key="semanticBuildButton">${currentTranslations.semanticBuildButton || 'BUILD INDEX'}</span>`;
        }, 3000);
    }
}

export async function updateIndexingProgress(once = false) {
    try {
        const response = await fetch('/api/semantic/status');
        const data = await response.json();
        semanticSearchSection.classList.toggle('hidden', !data.index_ready);
        if (data.status === 'running') {
            startIndexingButton.disabled = true;
            const progressText = data.total > 0 ? `(${data.progress} / ${data.total})` : '';
            startIndexingButton.innerHTML = `<i class="fas fa-spinner fa-spin"></i> INDEXING... ${progressText}`;
        } else if (data.status === 'complete' || data.status === 'error') {
            if (indexingPoller) { clearInterval(indexingPoller); indexingPoller = null; }
            startIndexingButton.disabled = false;
            let finalHTML;
            if (data.status === 'complete') {
                finalHTML = '<i class="fas fa-check"></i> INDEX READY';
                if (!once) { finishSound.play().catch(error => console.error("Audio playback failed:", error)); }
            } else {
                finalHTML = `<i class="fas fa-times"></i> ERROR`;
                startIndexingButton.title = data.current_file || 'An unknown error occurred.';
            }
            startIndexingButton.innerHTML = finalHTML;
            setTimeout(() => {
                if (!startIndexingButton.disabled) {
                    startIndexingButton.innerHTML = `<i class="fas fa-play-circle"></i> <span data-i18n-key="semanticBuildButton">${currentTranslations.semanticBuildButton || 'BUILD INDEX'}</span>`;
                    startIndexingButton.title = '';
                }
            }, 4000);
        }
    } catch (error) {
        if (indexingPoller) { clearInterval(indexingPoller); indexingPoller = null; }
        startIndexingButton.disabled = false;
        startIndexingButton.innerHTML = '<i class="fas fa-exclamation-triangle"></i> CONNECTION ERROR';
         setTimeout(() => {
            startIndexingButton.innerHTML = `<i class="fas fa-play-circle"></i> <span data-i18n-key="semanticBuildButton">${currentTranslations.semanticBuildButton || 'BUILD INDEX'}</span>`;
        }, 3000);
    }
}

export async function performSemanticSearch(event) {
    event.preventDefault();
    finishSound.play().catch(() => {}); finishSound.pause(); finishSound.currentTime = 0;
    const query = semanticQueryInput.value; if (!query) { return; }
    semanticResultsSection.classList.remove('hidden');
    semanticResultsDiv.innerHTML = '<div><i class="fas fa-spinner fa-spin"></i> Searching for meaning...</div>';

    lastSemanticResults = [];

    try {
        const response = await fetch('/api/semantic/search', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                k_fetch_initial: parseInt(quickKFetch.value, 10),
                vector_score_threshold: parseFloat(quickVectorScoreThreshold.value),
                vector_top_n: parseInt(quickVectorTopN.value, 10),
                enable_reranker: enableRerankerToggle.checked,
                rerank_score_threshold: parseFloat(quickScoreThreshold.value),
                rerank_top_n: parseInt(quickTopN.value, 10)
            })
        });
        if (!response.ok) {
            const err = await response.json();
            const error = new Error(err.detail);
            error.statusCode = response.status;
            throw error;
        }
        const results = await response.json();

        lastSemanticResults = results;

        renderSemanticResults(results);
        finishSound.play().catch(error => console.error("Audio playback failed:", error));
    } catch (error) {
        semanticResultsDiv.innerHTML = `<div class="status-error">Error: ${error.message}</div>`;
        if (error.statusCode === 409) {
            enableRerankerToggle.checked = false; enableRerankerToggle.disabled = true;
            enableRerankerSetting.checked = false; enableRerankerSetting.disabled = true;
            const label = document.querySelector('label[for="enable_reranker_toggle"]');
            if (label) label.title = "Enable in Settings and restart the application to use this feature.";
            rerankerOptionsDiv.classList.add('disabled');
        }
    }
}

function openChunkInNewTab(chunk, path) {
    const newTab = window.open();
    const htmlContent = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Chunk from ${escapeHTML(path)}</title>
        <style>body{font-family:sans-serif;line-height:1.6;background-color:#f0f4f8;color:#1a202c;padding:2rem;}h1{font-size:1.2rem;color:#5a6477;border-bottom:1px solid #ccc;padding-bottom:.5rem;margin-bottom:1rem;}pre{white-space:pre-wrap;word-wrap:break-word;background-color:#fff;padding:1.5rem;border-radius:6px;box-shadow:0 4px 15px rgba(26,32,44,.1);}</style>
        </head><body><h1>Source: ${escapeHTML(path)}</h1><pre>${escapeHTML(chunk)}</pre></body></html>`;
    newTab.document.write(htmlContent);
    newTab.document.close();
}

function renderSemanticResults(results) {
    semanticResultsDiv.innerHTML = '';
    if (!results || results.length === 0) {
        semanticResultsDiv.textContent = '> No relevant information found in the index.';
        return;
    }
    results.forEach((result, index) => {
        const item = document.createElement('div');
        item.className = 'semantic-result-item';
        let scoresHTML = '';
        if (result.vector_score !== undefined) {
            scoresHTML += `<span class="score-item" title="Vector Similarity Score"><span class="score-label">Vector:</span><span class="score-value">${result.vector_score.toFixed(4)}</span></span>`;
        }
        if (result.rerank_score !== undefined) {
            scoresHTML += `<span class="score-item" title="Rerank Relevance Score"><span class="score-label">Reranker:</span><span class="score-value">${result.rerank_score.toFixed(4)}</span></span>`;
        }
        item.innerHTML = `
            <div class="result-item-header">
                <span class="result-path">${escapeHTML(result.path)}</span>
                <div class="result-actions">
                    ${scoresHTML}
                    <i class="fas fa-folder-open action-btn" data-path="${escapeHTML(result.path)}" data-action="folder" title="Open Containing Folder"></i>
                    <i class="fas fa-external-link-alt action-btn action-btn-chunk" data-index="${index}" title="Open Chunk in New Tab"></i>
                </div>
            </div>
            <div class="result-chunk"><p>${escapeHTML(result.chunk)}</p></div>`;
        semanticResultsDiv.appendChild(item);
    });
}

export function initializeSemanticUI(defaults) {
    const defaultConfigValue = defaults?.retrieval_params?.enable_reranker || false;
    const savedPref = localStorage.getItem('enableReranker');
    const rerankerEnabled = savedPref !== null ? savedPref === 'true' : defaultConfigValue;
    enableRerankerToggle.checked = rerankerEnabled;
    enableRerankerToggle.disabled = false;
    enableRerankerSetting.checked = rerankerEnabled;
    enableRerankerSetting.disabled = false;
    rerankerOptionsDiv.classList.toggle('disabled', !rerankerEnabled);
}

export async function handleSemanticResultsClick(event) {
    const button = event.target.closest('.action-btn');
    if (!button) return;
    if (button.classList.contains('action-btn-chunk')) {
        const index = parseInt(button.dataset.index, 10);
        const result = lastSemanticResults[index];
        if (result) openChunkInNewTab(result.chunk, result.path);
    } else {
        await fetch('/api/open', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: button.dataset.path, action: button.dataset.action })
        });
    }
}


// 5. CLASSIFIER HANDLERS ########################################################################################
export async function startClassification() {
    finishSound.play().catch(() => {}); finishSound.pause(); finishSound.currentTime = 0;
    const path = classifierPathInput.value; if (!path) { alert("Please enter a directory path to classify."); return; }
    startClassificationButton.disabled = true;
    startClassificationButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> STARTING...';
    progressSection.classList.remove('hidden'); resultsSectionClassifier.classList.add('hidden'); resultsAccordion.innerHTML = '';
    try {
        const response = await fetch('/api/classifier/start', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ search_path: path })
        });
        if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'Failed to start classification.'); }
        classificationPoller = setInterval(updateClassificationProgress, 1000);
    } catch (error) {
        statusText.textContent = `Error: ${error.message}`;
        startClassificationButton.disabled = false;
        startClassificationButton.innerHTML = `<i class="fas fa-play-circle"></i> <span data-i18n-key="classifierStartButton">${currentTranslations.classifierStartButton || 'CLASSIFY FILES'}</span>`;
    }
}

async function updateClassificationProgress() {
    try {
        const response = await fetch('/api/classifier/status');
        const data = await response.json();
        if (data.status === 'running') {
            const percent = data.total > 0 ? (data.progress / data.total) * 100 : 0;
            progressBar.style.width = `${percent}%`; progressBar.textContent = `${Math.round(percent)}%`;
            statusText.textContent = `Scanning: ${data.current_file} (${data.progress} / ${data.total})`;
            startClassificationButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> CLASSIFYING...';
        } else {
            clearInterval(classificationPoller); classificationPoller = null;
            progressBar.style.width = '100%'; progressBar.textContent = '100%';
            statusText.textContent = data.current_file || 'Finished!';
            startClassificationButton.disabled = false;
            startClassificationButton.innerHTML = `<i class="fas fa-play-circle"></i> <span data-i18n-key="classifierStartButton">${currentTranslations.classifierStartButton || 'CLASSIFY FILES'}</span>`;
            if (data.status === 'complete') {
                renderClassificationResults();
                finishSound.play().catch(error => console.error("Audio playback failed:", error));
            } else if (data.status === 'error') { statusText.textContent = `Error: ${data.current_file || 'An unknown error occurred.'}`; }
        }
    } catch (error) { clearInterval(classificationPoller); classificationPoller = null; statusText.textContent = 'Error: Could not get status from server.'; }
}

export async function renderClassificationResults() {
    try {
        const response = await fetch('/api/classifier/results');
        const data = await response.json();
        resultsAccordion.innerHTML = '';

        if (Object.keys(data).length === 0) {
            resultsSectionClassifier.classList.add('hidden');
            organizeAllButton.classList.add('hidden');
            return;
        }
        resultsSectionClassifier.classList.remove('hidden');
        organizeAllButton.classList.remove('hidden');

        for (const tag of Object.keys(data).sort()) {
            const files = data[tag]; const item = document.createElement('div');
            item.className = 'accordion-item';
            item.innerHTML = `
                <div class="accordion-header">
                    <h3><i class="fas fa-tag"></i> ${escapeHTML(tag)}</h3>
                    <div class="header-actions">
                        <button class="icon-btn file-op-btn" data-tag="${escapeHTML(tag)}" data-action="copy" title="Copy Files"><i class="fas fa-copy"></i></button>
                        <button class="icon-btn file-op-btn danger" data-tag="${escapeHTML(tag)}" data-action="move" title="Move Files"><i class="fas fa-cut"></i></button>
                        <button class="icon-btn delete-tag-btn danger" data-tag="${escapeHTML(tag)}" title="Delete Classification Entry"><i class="fas fa-trash-alt"></i></button>
                        <button class="icon-btn export-btn" data-tag="${escapeHTML(tag)}" title="Export List"><i class="fas fa-file-export"></i></button>
                        <span class="tag-count">${files.length} files</span>
                    </div>
                </div>
                <div class="accordion-content">
                    ${files.map(path => `<div class="result-item"><span class="result-path">${escapeHTML(path)}</span><div class="result-actions"><i class="fas fa-folder-open action-btn" data-path="${escapeHTML(path)}" data-action="folder" title="Open Containing Folder"></i></div></div>`).join('')}
                </div>
            `;
            resultsAccordion.appendChild(item);
        }
    } catch (error) { console.error("Failed to render classification results:", error); }
}

function exportCategory(tag, files) {
    const content = files.join('\n'); const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = `classified_${tag}_files.txt`; document.body.appendChild(a);
    a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
}

function getDestinationPathFromModal(tag, action) {
    return new Promise((resolve) => {
        modalTitle.textContent = `${action.charAt(0).toUpperCase() + action.slice(1)} Files`;
        modalText.textContent = `Please enter the full destination path to ${action} all files tagged as "${tag}":`;
        modalInput.value = ''; modalErrorText.textContent = ''; modalErrorText.classList.add('hidden');
        modalInput.classList.remove('input-error'); modalOverlay.classList.remove('hidden'); modalInput.focus();
        const cleanupAndResolve = (value) => {
            modalConfirmBtn.onclick = null; modalCancelBtn.onclick = null; modalCloseBtn.onclick = null;
            document.removeEventListener('keydown', keydownHandler);
            modalOverlay.classList.add('hidden'); resolve(value);
        };
        const confirmAction = () => {
            const path = modalInput.value.trim();
            if (path) { cleanupAndResolve(path); } else {
                modalErrorText.textContent = 'Please provide a destination path.'; modalErrorText.classList.remove('hidden');
                modalInput.classList.add('input-error'); setTimeout(() => modalInput.classList.remove('input-error'), 820);
            }
        };
        const cancelAction = () => { cleanupAndResolve(null); };
        const keydownHandler = (event) => {
            if (event.key === 'Enter') { event.preventDefault(); confirmAction(); } else if (event.key === 'Escape') { cancelAction(); }
        };
        modalConfirmBtn.onclick = confirmAction; modalCancelBtn.onclick = cancelAction; modalCloseBtn.onclick = cancelAction;
        document.addEventListener('keydown', keydownHandler);
    });
}

async function handleFileOperation(button) {
    const tag = button.dataset.tag; const action = button.dataset.action;
    const destinationPath = await getDestinationPathFromModal(tag, action);
    if (!destinationPath) { return; }
    const originalIcon = button.innerHTML; button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>'; button.disabled = true;
    try {
        const response = await fetch('/api/classifier/file-operation', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tag, action, destination_path: destinationPath })
        });
        const result = await response.json();
        if (!response.ok) { throw new Error(result.detail || 'An unknown error occurred.'); }
        alert(result.message);
        if (action === 'move') { renderClassificationResults(); }
    } catch (error) { alert(`Error: ${error.message}`); } finally { button.innerHTML = originalIcon; button.disabled = false; }
}

async function handleDeleteClassification(button) {
    const tag = button.dataset.tag;
    const confirmationMessage = currentTranslations.deleteClassificationConfirmation || `Are you sure you want to delete all classification results for the tag "${tag}"? This will not delete the files from your disk, but the entry will be removed from this list.`;

    if (confirm(confirmationMessage)) {
        const originalIcon = button.innerHTML; button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>'; button.disabled = true;
        try {
            const response = await fetch('/api/classifier/delete-tag', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tag: tag })
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Failed to delete the classification entry.');
            await renderClassificationResults();
        } catch (error) {
            alert(`Error: ${error.message}`);
            button.innerHTML = originalIcon; button.disabled = false;
        }
    }
}

export async function handleOrganizeAll() {
    const baseDestinationPath = await getDestinationPathFromModal("All Categories", "organize");
    if (!baseDestinationPath) { return; }
    organizeAllButton.disabled = true;
    organizeAllButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Organizing...';
    try {
        const response = await fetch('/api/classifier/organize-all', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ base_destination_path: baseDestinationPath })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'An unknown error occurred.');
        alert(result.message);
        await renderClassificationResults();
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        organizeAllButton.disabled = false;
        const btnSpan = document.createElement('span');
        btnSpan.dataset.i18nKey = 'organizeAllButton';
        btnSpan.textContent = currentTranslations.organizeAllButton || 'Auto Organize';
        organizeAllButton.innerHTML = `<i class="fas fa-folder-tree"></i> `;
        organizeAllButton.appendChild(btnSpan);
    }
}

export async function handleClassifierAccordionClick(event) {
    const header = event.target.closest('.accordion-header');
    const exportBtn = event.target.closest('.export-btn');
    const actionBtn = event.target.closest('.action-btn');
    const fileOpBtn = event.target.closest('.file-op-btn');
    const deleteTagBtn = event.target.closest('.delete-tag-btn');

    if (deleteTagBtn) {
        event.stopPropagation(); await handleDeleteClassification(deleteTagBtn);
    } else if (fileOpBtn) {
        event.stopPropagation(); await handleFileOperation(fileOpBtn);
    } else if (exportBtn) {
        event.stopPropagation();
        const tag = exportBtn.dataset.tag;
        const response = await fetch('/api/classifier/results');
        const data = await response.json();
        if (data[tag]) exportCategory(tag, data[tag]);
    } else if (header) {
        header.nextElementSibling.classList.toggle('active');
    } else if (actionBtn) {
        await fetch('/api/open', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: actionBtn.dataset.path, action: actionBtn.dataset.action })
        });
    }
}


// 6. CLASSIFIER TRAINER HANDLERS ################################################################################
export async function startTraining() {
    const path = trainingDataPathInput.value;
    if (!path) { alert("Please provide the path to the training data directory."); return; }
    startTrainingButton.disabled = true; startTrainingButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> STARTING...';
    trainerStatusSection.classList.remove('hidden'); trainerLog.textContent = 'Initializing request...';
    trainerResult.textContent = '';
    try {
        const response = await fetch('/api/trainer/start', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data_path: path, test_size: parseFloat(testSizeSlider.value), n_estimators: parseInt(nEstimatorsInput.value, 10) })
        });
        if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'Failed to start training task.'); }
        if (trainingPoller) clearInterval(trainingPoller);
        trainingPoller = setInterval(updateTrainingProgress, 2000);
    } catch (error) {
        trainerLog.textContent = `Error: ${error.message}`;
        startTrainingButton.disabled = false;
        startTrainingButton.innerHTML = `<i class="fas fa-play-circle"></i> <span data-i18n-key="settingsTrainerStartButton">${currentTranslations.settingsTrainerStartButton || 'START TRAINING'}</span>`;
    }
}

async function updateTrainingProgress() {
    try {
        const response = await fetch('/api/trainer/status');
        const data = await response.json();
        trainerLog.textContent = (data.log || []).join('\n'); trainerLog.scrollTop = trainerLog.scrollHeight;
        if (data.status === 'running') {
            startTrainingButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> TRAINING IN PROGRESS...';
        } else {
            if (trainingPoller) clearInterval(trainingPoller); trainingPoller = null;
            startTrainingButton.disabled = false;
            startTrainingButton.innerHTML = `<i class="fas fa-play-circle"></i> <span data-i18n-key="settingsTrainerStartButton">${currentTranslations.settingsTrainerStartButton || 'START TRAINING'}</span>`;
            if (data.status === 'complete') {
                trainerResult.textContent = `✅ Training Complete! Final Accuracy: ${data.accuracy}`;
                trainerResult.style.color = 'var(--color-accent-secondary)';
            } else if (data.status === 'error') {
                trainerResult.textContent = `❌ An error occurred. Please check the log above.`;
                trainerResult.style.color = 'var(--color-danger)';
            }
        }
    } catch (error) {
        if (trainingPoller) clearInterval(trainingPoller);
        trainerResult.textContent = 'Error: Could not get status from server.';
        trainerResult.style.color = 'var(--color-danger)';
    }
}
