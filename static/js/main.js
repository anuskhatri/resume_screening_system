/**
 * AI Resume Screener - Interactive Client Suite
 * Includes: Drag-and-drop file ingestion, Async NLP JD Parsing,
 * Live search/filtering, Candidate Detail Modal, and Side-by-Side Comparator.
 */

document.addEventListener('DOMContentLoaded', () => {
    initWeightControls();
    initJDUpload();
    initResumeUpload();
    initResultsFiltering();
    initCandidateModal();
    initCandidateComparator();
    initScreeningFormSubmit();
});

function initWeightControls() {
    document.querySelectorAll('.weight-control').forEach(group => {
        const inputs = [...group.querySelectorAll('input[type="range"]')];
        const update = () => {
            const total = inputs.reduce((sum, input) => sum + Number(input.value), 0) || 1;
            inputs.forEach(input => {
                const output = group.querySelector(`[data-weight-label="${input.name.replace('_weight', '')}"]`);
                if (output) output.textContent = `${Math.round(Number(input.value) / total * 100)}%`;
            });
        };
        inputs.forEach(input => input.addEventListener('input', update));
        update();
    });
}

/* ==========================================================================
   1. Job Description File Ingestion & Async NLP Parsing
   ========================================================================== */
function initJDUpload() {
    const jdDropzone = document.getElementById('jd_dropzone');
    const jdFileInput = document.getElementById('jd_file');
    const jdTextarea = document.getElementById('job_description');
    const jdParsedContainer = document.getElementById('jd_parsed_preview');
    const jdStatusText = document.getElementById('jd_status_text');

    if (!jdDropzone || !jdFileInput) return;

    jdDropzone.addEventListener('click', () => jdFileInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
        jdDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            jdDropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        jdDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            jdDropzone.classList.remove('dragover');
        });
    });

    jdDropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            jdFileInput.files = e.dataTransfer.files;
            handleJDParsing(e.dataTransfer.files[0]);
        }
    });

    jdFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleJDParsing(e.target.files[0]);
        }
    });

    // Also support manual "Extract from Text" button if user pasted text
    const btnExtractText = document.getElementById('btn_extract_jd_text');
    if (btnExtractText && jdTextarea) {
        btnExtractText.addEventListener('click', () => {
            const text = jdTextarea.value.trim();
            if (text.length < 15) {
                alert('Please enter or paste a more detailed Job Description first.');
                return;
            }
            sendJDParseRequest({ job_description: text });
        });
    }

    function handleJDParsing(file) {
        if (jdStatusText) {
            jdStatusText.innerHTML = `<span class="spinner-border spinner-border-sm text-primary me-2"></span> Analyzing <strong>${file.name}</strong> with NLP...`;
        }

        const formData = new FormData();
        formData.append('jd_file', file);
        sendJDParseRequest(formData, true);
    }

    function sendJDParseRequest(data, isFormData = false) {
        const options = {
            method: 'POST',
            body: isFormData ? data : JSON.stringify(data),
            headers: isFormData ? {} : { 'Content-Type': 'application/json' }
        };

        fetch('/api/parse-jd', options)
            .then(res => {
                if (!res.ok) throw new Error('Could not parse Job Description');
                return res.json();
            })
            .then(data => {
                if (jdTextarea && data.raw_text) {
                    jdTextarea.value = data.raw_text;
                }

                // Auto-fill experience and education inputs
                const expInput = document.getElementById('required_experience');
                if (expInput && data.required_experience !== undefined) {
                    expInput.value = data.required_experience;
                }

                const eduInput = document.getElementById('required_education');
                if (eduInput && data.required_education) {
                    eduInput.value = data.required_education;
                }

                if (jdStatusText) {
                    jdStatusText.innerHTML = `<i class="fas fa-check-circle text-success me-1"></i> NLP Extraction Complete: <strong>${data.job_title}</strong> (${data.seniority})`;
                }

                // Render badges preview
                if (jdParsedContainer) {
                    renderParsedJDBadges(data, jdParsedContainer);
                }
            })
            .catch(err => {
                console.error('JD Parsing Error:', err);
                if (jdStatusText) {
                    jdStatusText.innerHTML = `<span class="text-warning"><i class="fas fa-exclamation-circle me-1"></i> ${err.message}. You can manually edit requirements below.</span>`;
                }
            });
    }
}

function renderParsedJDBadges(data, container) {
    container.style.display = 'block';
    let skillsHtml = '';
    if (data.core_skills && data.core_skills.length > 0) {
        skillsHtml += `<div class="mb-2"><strong class="small text-secondary">Core Skills:</strong><br>`;
        data.core_skills.forEach(s => {
            skillsHtml += `<span class="tag-pill tag-pill-primary"><i class="fas fa-check-circle text-primary"></i> ${s}</span>`;
        });
        skillsHtml += `</div>`;
    }

    if (data.preferred_skills && data.preferred_skills.length > 0) {
        skillsHtml += `<div class="mb-2"><strong class="small text-secondary">Preferred / Bonus:</strong><br>`;
        data.preferred_skills.forEach(s => {
            skillsHtml += `<span class="tag-pill tag-pill-cyan"><i class="fas fa-star text-info"></i> ${s}</span>`;
        });
        skillsHtml += `</div>`;
    }

    container.innerHTML = `
        <div class="d-flex align-items-center justify-content-between mb-3 border-bottom pb-2 border-secondary border-opacity-25">
            <div>
                <span class="badge bg-primary bg-opacity-25 text-primary border border-primary border-opacity-50 me-2">${data.seniority}</span>
                <strong class="text-white fs-6">${data.job_title}</strong>
            </div>
            <div class="text-end">
                <span class="tag-pill tag-pill-emerald"><i class="fas fa-clock"></i> ${data.required_experience}+ Years Exp</span>
            </div>
        </div>
        ${skillsHtml}
        <div class="small text-muted mt-2"><i class="fas fa-graduation-cap me-1"></i> <strong>Education:</strong> ${data.required_education}</div>
    `;
}

/* ==========================================================================
   2. Resumes Multi-File Drag-and-Drop
   ========================================================================== */
function initResumeUpload() {
    const resumesDropzone = document.getElementById('resumes_dropzone');
    const resumesInput = document.getElementById('resumes');
    const chipsWrapper = document.getElementById('resume_chips_wrapper');
    const countBadge = document.getElementById('resume_count_badge');

    if (!resumesDropzone || !resumesInput) return;

    resumesDropzone.addEventListener('click', () => resumesInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
        resumesDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            resumesDropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        resumesDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            resumesDropzone.classList.remove('dragover');
        });
    });

    resumesDropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            resumesInput.files = e.dataTransfer.files;
            updateResumeChips(resumesInput.files);
        }
    });

    resumesInput.addEventListener('change', (e) => {
        updateResumeChips(e.target.files);
    });

    function updateResumeChips(files) {
        if (!chipsWrapper) return;
        chipsWrapper.innerHTML = '';
        if (!files || files.length === 0) {
            if (countBadge) countBadge.textContent = '0 files';
            return;
        }

        if (countBadge) {
            countBadge.textContent = `${files.length} file${files.length > 1 ? 's' : ''} staged`;
        }

        Array.from(files).forEach((file, idx) => {
            const sizeKb = Math.round(file.size / 1024);
            const ext = file.name.split('.').pop().toUpperCase();
            const chip = document.createElement('div');
            chip.className = 'file-chip';
            chip.innerHTML = `
                <i class="fas fa-file-alt text-primary"></i>
                <span class="text-truncate" style="max-width: 180px;">${file.name}</span>
                <span class="badge bg-secondary bg-opacity-50 text-white-50 ms-1">${ext} · ${sizeKb}KB</span>
            `;
            chipsWrapper.appendChild(chip);
        });
    }
}

/* ==========================================================================
   3. Results Filtering, Searching & Sorting
   ========================================================================== */
function initResultsFiltering() {
    const searchInput = document.getElementById('candidate_search');
    const statusFilter = document.getElementById('filter_status');
    const tierFilter = document.getElementById('filter_tier');
    const sortSelect = document.getElementById('sort_by');
    const candidateCards = document.querySelectorAll('.candidate-card-item');

    if (!candidateCards.length) return;

    function applyFilters() {
        const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
        const statusVal = statusFilter ? statusFilter.value : 'all';
        const tierVal = tierFilter ? tierFilter.value : 'all';

        let visibleCount = 0;

        candidateCards.forEach(card => {
            const name = card.dataset.name ? card.dataset.name.toLowerCase() : '';
            const skills = card.dataset.skills ? card.dataset.skills.toLowerCase() : '';
            const shortlisted = card.dataset.shortlisted === 'true';
            const tier = card.dataset.tier ? card.dataset.tier.toLowerCase() : '';

            const matchesQuery = !query || name.includes(query) || skills.includes(query);
            const matchesStatus = statusVal === 'all' || 
                (statusVal === 'shortlisted' && shortlisted) || 
                (statusVal === 'unselected' && !shortlisted);
            const matchesTier = tierVal === 'all' || tier === tierVal.toLowerCase();

            if (matchesQuery && matchesStatus && matchesTier) {
                card.style.display = '';
                visibleCount++;
            } else {
                card.style.display = 'none';
            }
        });

        const noResultsAlert = document.getElementById('no_filter_results');
        if (noResultsAlert) {
            noResultsAlert.style.display = visibleCount === 0 ? 'block' : 'none';
        }
    }

    if (searchInput) searchInput.addEventListener('input', applyFilters);
    if (statusFilter) statusFilter.addEventListener('change', applyFilters);
    if (tierFilter) tierFilter.addEventListener('change', applyFilters);

    if (sortSelect) {
        sortSelect.addEventListener('change', () => {
            const container = document.getElementById('candidates_container');
            if (!container) return;
            const sortVal = sortSelect.value;
            const cardsArray = Array.from(candidateCards);

            cardsArray.sort((a, b) => {
                if (sortVal === 'score_desc') return parseFloat(b.dataset.score) - parseFloat(a.dataset.score);
                if (sortVal === 'score_asc') return parseFloat(a.dataset.score) - parseFloat(b.dataset.score);
                if (sortVal === 'exp_desc') return parseFloat(b.dataset.exp) - parseFloat(a.dataset.exp);
                if (sortVal === 'rank_asc') return parseInt(a.dataset.rank) - parseInt(b.dataset.rank);
                return 0;
            });

            cardsArray.forEach(card => container.appendChild(card));
        });
    }
}

/* ==========================================================================
   4. Candidate Deep-Dive Modal
   ========================================================================== */
function initCandidateModal() {
    const modalEl = document.getElementById('candidateDetailModal');
    if (!modalEl) return;

    modalEl.addEventListener('show.bs.modal', function (event) {
        const button = event.relatedTarget;
        if (!button) return;

        const dataStr = button.getAttribute('data-candidate');
        if (!dataStr) return;

        try {
            const cand = JSON.parse(dataStr);
            document.getElementById('modal_cand_name').textContent = cand.filename || 'Candidate Details';
            document.getElementById('modal_cand_rank').textContent = `#${cand.rank || '1'}`;
            document.getElementById('modal_cand_score').textContent = `${Math.round(cand.final_score * 100)}%`;
            document.getElementById('modal_cand_tier').textContent = cand.match_tier || 'Strong Fit';
            document.getElementById('modal_cand_exp').textContent = `${cand.experience_years || 0} Years`;
            document.getElementById('modal_cand_edu').textContent = cand.candidate_education || 'Bachelor\'s Degree';

            // Progress bars
            setModalProgressBar('modal_bar_semantic', cand.semantic_score);
            setModalProgressBar('modal_bar_skill', cand.skill_score || (cand.skill_match_pct / 100));
            setModalProgressBar('modal_bar_exp', cand.experience_score);
            setModalProgressBar('modal_bar_edu', cand.education_score);
            const requirements = cand.requirements || {};
            setRequirementStatus('modal_requirement_skills', requirements.skills, `${requirements.skills?.matched || 0} / ${requirements.skills?.required || 0} core skills`);
            setRequirementStatus('modal_requirement_experience', requirements.experience, `${requirements.experience?.candidate ?? 0} / ${requirements.experience?.required ?? 0} years`, !requirements.experience?.evidence);
            const educationEvidence = requirements.education?.candidate && requirements.education.candidate !== 'Not Specified';
            setRequirementStatus('modal_requirement_education', requirements.education, educationEvidence ? requirements.education.candidate : 'No degree evidence found', !educationEvidence);
            const requirementEvidence = document.getElementById('modal_requirement_evidence');
            if (requirementEvidence) {
                requirementEvidence.replaceChildren();
                [['Experience', requirements.experience?.evidence], ['Education', requirements.education?.evidence]].forEach(([label, sentence]) => {
                    if (!sentence) return;
                    const li = document.createElement('li');
                    const strong = document.createElement('strong');
                    strong.textContent = `${label}: `;
                    li.append(strong, document.createTextNode(sentence));
                    requirementEvidence.appendChild(li);
                });
            }

            // Skill Gap
            renderSkillPills('modal_matched_skills', cand.matched_skills, 'tag-pill-emerald', 'fa-check');
            renderSkillPills('modal_missing_skills', cand.missing_skills, 'tag-pill-rose', 'fa-times');
            renderSkillPills('modal_bonus_skills', cand.bonus_skills, 'tag-pill-cyan', 'fa-plus');

            const evidenceList = document.getElementById('modal_skill_evidence');
            if (evidenceList) {
                evidenceList.replaceChildren();
                const evidence = cand.skill_evidence || {};
                Object.entries(evidence).forEach(([skill, snippets]) => snippets.forEach(snippet => {
                    const li = document.createElement('li');
                    const label = document.createElement('strong');
                    label.textContent = `${skill}: `;
                    li.append(label, document.createTextNode(snippet));
                    evidenceList.appendChild(li);
                }));
                if (!evidenceList.children.length) evidenceList.innerHTML = '<li class="text-muted">No direct supporting sentence found for the detected skills.</li>';
            }

            // Highlights
            const highlightsList = document.getElementById('modal_highlights_list');
            if (highlightsList) {
                highlightsList.innerHTML = '';
                if (cand.highlights && cand.highlights.length > 0) {
                    cand.highlights.forEach(h => {
                        const li = document.createElement('li');
                        li.className = 'mb-2 text-light';
                        li.textContent = h;
                        highlightsList.appendChild(li);
                    });
                } else {
                    highlightsList.innerHTML = '<li class="text-muted">No specific highlights extracted.</li>';
                }
            }

            // Preview Text
            const previewEl = document.getElementById('modal_resume_preview');
            if (previewEl) {
                previewEl.textContent = cand.resume_preview || 'Resume preview not available.';
            }
        } catch (e) {
            console.error('Failed to parse candidate JSON:', e);
        }
    });

    function setModalProgressBar(id, value) {
        const bar = document.getElementById(id);
        const valSpan = document.getElementById(`${id}_val`);
        const pct = Math.round((value || 0) * 100);
        if (bar) bar.style.width = `${pct}%`;
        if (valSpan) valSpan.textContent = `${pct}%`;
    }

    function setRequirementStatus(id, criterion, evidence, missingEvidence = false) {
        const target = document.getElementById(id);
        if (!target) return;
        const status = missingEvidence ? 'Review' : (criterion?.meets === true ? 'Meets' : (criterion?.meets === false ? 'Below' : 'Review'));
        target.textContent = `${status} · ${evidence}`;
        target.dataset.status = status.toLowerCase();
    }

    function renderSkillPills(id, skills, pillClass, iconClass) {
        const el = document.getElementById(id);
        if (!el) return;
        el.innerHTML = '';
        if (skills && skills.length > 0) {
            skills.forEach(s => {
                const span = document.createElement('span');
                span.className = `tag-pill ${pillClass}`;
                span.innerHTML = `<i class="fas ${iconClass} me-1"></i> ${s}`;
                el.appendChild(span);
            });
        } else {
            el.innerHTML = '<span class="text-muted small">None detected</span>';
        }
    }
}

/* ==========================================================================
   5. Side-by-Side Candidate Comparator
   ========================================================================== */
function initCandidateComparator() {
    const drawer = document.getElementById('comparison_drawer');
    const selectedCountSpan = document.getElementById('compare_selected_count');
    const btnOpenCompareModal = document.getElementById('btn_open_compare_modal');
    const btnClearCompare = document.getElementById('btn_clear_compare');
    const checkboxes = document.querySelectorAll('.compare-checkbox');

    if (!drawer || !checkboxes.length) return;

    let selectedCandidates = [];

    checkboxes.forEach(cb => {
        cb.addEventListener('change', () => {
            const candData = JSON.parse(cb.getAttribute('data-candidate'));
            if (cb.checked) {
                if (selectedCandidates.length >= 3) {
                    alert('You can compare a maximum of 3 candidates simultaneously.');
                    cb.checked = false;
                    return;
                }
                selectedCandidates.push(candData);
            } else {
                selectedCandidates = selectedCandidates.filter(c => c.resume_id !== candData.resume_id);
            }

            updateDrawerState();
        });
    });

    if (btnClearCompare) {
        btnClearCompare.addEventListener('click', () => {
            selectedCandidates = [];
            checkboxes.forEach(cb => cb.checked = false);
            updateDrawerState();
        });
    }

    function updateDrawerState() {
        if (selectedCountSpan) selectedCountSpan.textContent = selectedCandidates.length;
        if (selectedCandidates.length >= 2) {
            drawer.classList.add('active');
        } else {
            drawer.classList.remove('active');
        }
    }

    if (btnOpenCompareModal) {
        btnOpenCompareModal.addEventListener('click', () => {
            renderComparisonTable(selectedCandidates);
            const compModal = new bootstrap.Modal(document.getElementById('comparatorModal'));
            compModal.show();
        });
    }

    function renderComparisonTable(candidates) {
        const thead = document.getElementById('compare_table_head');
        const tbody = document.getElementById('compare_table_body');
        if (!thead || !tbody) return;

        // Render header
        let headHtml = `<tr><th style="width: 200px;">Metric / Attribute</th>`;
        candidates.forEach(c => {
            headHtml += `<th class="text-center">
                <span class="badge bg-primary bg-opacity-25 text-primary mb-1">Rank #${c.rank}</span><br>
                <strong class="text-white">${c.filename}</strong>
            </th>`;
        });
        headHtml += `</tr>`;
        thead.innerHTML = headHtml;

        // Render metric rows
        const rows = [
            { label: 'Overall Score', key: c => `<strong class="fs-5 text-primary">${Math.round(c.final_score * 100)}%</strong>` },
            { label: 'Fit Tier', key: c => `<span class="badge ${c.final_score >= 0.7 ? 'bg-success' : 'bg-warning'}">${c.match_tier}</span>` },
            { label: 'Semantic Match', key: c => `${Math.round(c.semantic_score * 100)}%` },
            { label: 'Skill Match %', key: c => `${c.skill_match_pct}%` },
            { label: 'Experience (Years)', key: c => `<strong>${c.experience_years} Years</strong>` },
            { label: 'Education Level', key: c => c.candidate_education || 'Bachelor\'s' },
            {
                label: 'Matched Skills',
                key: c => c.matched_skills.map(s => `<span class="tag-pill tag-pill-emerald">${s}</span>`).join(' ') || '<span class="text-muted">None</span>'
            },
            {
                label: 'Missing Skills',
                key: c => c.missing_skills.map(s => `<span class="tag-pill tag-pill-rose">${s}</span>`).join(' ') || '<span class="text-muted">None</span>'
            }
        ];

        let bodyHtml = '';
        rows.forEach(r => {
            bodyHtml += `<tr><td class="fw-semibold text-secondary">${r.label}</td>`;
            candidates.forEach(c => {
                bodyHtml += `<td class="text-center align-middle">${r.key(c)}</td>`;
            });
            bodyHtml += `</tr>`;
        });
        tbody.innerHTML = bodyHtml;
    }
}

/* ==========================================================================
   6. Progressive Screening Form Submit Stepper
   ========================================================================== */
function initScreeningFormSubmit() {
    const uploadForm = document.getElementById('screening_upload_form');
    const overlay = document.getElementById('processing_overlay');

    if (!uploadForm || !overlay) return;

    uploadForm.addEventListener('submit', () => {
        overlay.classList.add('active');
        const steps = [
            { id: 'step_1', text: 'Extracting Resume & JD text...', delay: 800 },
            { id: 'step_2', text: 'SpaCy Entity & 500+ Skill Ontology Mining...', delay: 2400 },
            { id: 'step_3', text: 'Sentence-BERT GPU Semantic Vectorization...', delay: 4200 },
            { id: 'step_4', text: 'Calculating Skill Gap & Ranking Candidates...', delay: 6000 }
        ];

        steps.forEach(s => {
            setTimeout(() => {
                const stepEl = document.getElementById(s.id);
                if (stepEl) {
                    stepEl.classList.remove('text-muted');
                    stepEl.classList.add('text-primary', 'fw-bold');
                    stepEl.innerHTML = `<i class="fas fa-check-circle text-success me-2"></i> ${s.text}`;
                }
            }, s.delay);
        });
    });
}
