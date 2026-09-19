let steps = [];
    let currentIndex = 0;
    let generatedText = "";
    let autoTimer = null;
    let chatHistory = [];
    let activeAssistantBubble = null;
    let selectedFiles = [];
    const $ = (id) => document.getElementById(id);

    function setTheme(theme) {
      document.body.dataset.theme = theme;
      localStorage.setItem('nexus-theme', theme);
      const btn = $('theme-toggle');
      if (btn) btn.textContent = theme === 'dark' ? '🌙' : '☀️';
    }
    function toggleTheme() {
      setTheme(document.body.dataset.theme === 'dark' ? 'light' : 'dark');
    }

    function openDrawer(tab=null) {
      $('drawer').classList.add('open');
      $('overlay').classList.add('open');
      $('app').classList.add('drawer-shift');
      if (tab) showTab(tab);
    }
    function closeDrawer() {
      $('drawer').classList.remove('open');
      $('overlay').classList.remove('open');
      $('app').classList.remove('drawer-shift');
    }
    function showTab(name) {
      for (const t of ['generation','rag','ner']) {
        $(`tab-${t}`).classList.toggle('active', t === name);
        $(`tab-${t}-btn`).classList.toggle('active', t === name);
      }
    }

    function syncControls() { $('temperature-value').textContent = $('temperature').value; $('topk-value').textContent = $('top-k').value; }
    function setLoading(active, msg='') { $('loading').classList.toggle('active', active); $('generate-btn').disabled = active; }

    function addMessage(role, text, meta=null) {
      $('empty-state').style.display = 'none';
      const row = document.createElement('div');
      row.className = `message ${role}`;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      if (meta) {
        const m = document.createElement('span');
        m.className = 'bubble-meta';
        m.textContent = meta;
        bubble.appendChild(m);
      }
      const content = document.createElement('span');
      content.className = 'bubble-content';
      content.textContent = text;
      bubble.appendChild(content);
      row.appendChild(bubble);
      $('chat-log').appendChild(row);
      $('response-area').scrollTop = $('response-area').scrollHeight;
      return content;
    }

    function resetGenerationPanel() {
      steps=[]; currentIndex=0; generatedText=''; activeAssistantBubble=null;
      if(autoTimer) clearInterval(autoTimer); autoTimer=null;
      $('current-token').textContent='Aucun token';
      $('alternatives').innerHTML='<p class="notice">Les top tokens apparaîtront ici après génération.</p>';
      $('rag-results').innerHTML='<p class="notice">Active le RAG et ajoute un fichier pour voir les passages utilisés.</p>';
      $('ner-results').innerHTML='<p class="notice">Active le NER pour afficher les entités Chemical/Disease détectées.</p>';
      $('next-btn').disabled=true; $('auto-btn').disabled=true; $('auto-btn').textContent='Lecture auto'; resetMetrics();
    }

    function resetMetrics() {
      $('m-prompt').textContent='0'; $('m-generated').textContent='0'; $('m-remaining').textContent='0'; $('m-context').textContent='0'; $('m-progress').textContent='0 %'; $('m-time').textContent='0 s'; $('m-speed').textContent='0 tok/s'; $('progress-bar').style.width='0%';
    }

    function resetAll() {
      resetGenerationPanel();
      chatHistory=[];
      $('chat-log').innerHTML='';
      $('empty-state').style.display='grid';
      clearAttachments();
    }

    function updateMetrics(m) {
      if(!m) return; $('m-prompt').textContent=m.prompt_tokens??0; $('m-generated').textContent=m.shown_tokens??0; $('m-remaining').textContent=m.remaining_tokens??0; $('m-context').textContent=m.current_context_size??0; $('m-progress').textContent=`${(m.progress??0).toFixed(0)} %`; $('m-time').textContent=`${(m.elapsed_seconds??0).toFixed(2)} s`; $('m-speed').textContent=`${(m.tokens_per_second??0).toFixed(2)} tok/s`; $('progress-bar').style.width=`${Math.min(100,m.progress??0)}%`;
    }
    function renderAlternatives(alts) {
      if(!alts||!alts.length){ $('alternatives').innerHTML='<p class="notice">Aucune alternative disponible.</p>'; return; }
      $('alternatives').innerHTML = alts.map(a=>`<div class="item ${a.chosen?'chosen':''}"><div class="row"><span class="mono">${escapeHtml(a.token)}</span><span class="score">${(a.prob*100).toFixed(2)} %</span></div></div>`).join('');
    }
    function renderRagResults(chunks) {
      if(!chunks||!chunks.length){ $('rag-results').innerHTML='<p class="notice">Aucun passage RAG utilisé.</p>'; return; }
      $('rag-results').innerHTML = chunks.map((c,i)=>`<div class="item"><div class="row"><b>#${i+1} ${escapeHtml(c.filename)}</b><span class="score">${Number(c.score).toFixed(3)}</span></div><div style="margin-top:5px">${escapeHtml(c.text.slice(0,260))}${c.text.length>260?'...':''}</div></div>`).join('');
    }
    function renderNerResults(entities, error) {
      if(error){ $('ner-results').innerHTML=`<p class="notice">${escapeHtml(error)}</p>`; return; }
      if(!entities||!entities.length){ $('ner-results').innerHTML='<p class="notice">Aucune entité détectée ou NER non activé.</p>'; return; }
      $('ner-results').innerHTML = entities.map(e=>`<div class="item"><span class="badge">${escapeHtml(e.entity_group)}</span><b>${escapeHtml(e.word)}</b><div class="row" style="margin-top:5px"><span class="score">score ${(e.score*100).toFixed(1)} %</span><span class="score">x${e.count||1}</span></div></div>`).join('');
    }
    function activeModeLabel(useRag, useNer, useOllama) {
      const parts = [];
      parts.push(useOllama ? 'Ollama' : 'NexusLM');
      if(useRag) parts.push('RAG');
      if(useNer) parts.push('NER');
      return parts.join(' + ');
    }

    function showNextToken() {
      if(currentIndex>=steps.length) return;
      const step=steps[currentIndex]; const token=step.token; generatedText += (generatedText?' ':'') + token;
      if (!activeAssistantBubble) activeAssistantBubble = addMessage('assistant', '', 'NexusLM');
      activeAssistantBubble.innerHTML = `${escapeHtml(generatedText)} <span class="token-new">${escapeHtml(token)}</span>`;
      setTimeout(()=>{ if(activeAssistantBubble) activeAssistantBubble.textContent=generatedText; }, 220);
      $('current-token').textContent=token; renderAlternatives(step.alternatives); updateMetrics(step.metrics); currentIndex += 1;
      $('response-area').scrollTop=$('response-area').scrollHeight;
      if(currentIndex>=steps.length){
        $('next-btn').disabled=true; $('auto-btn').disabled=true;
        if(autoTimer) clearInterval(autoTimer); autoTimer=null; $('auto-btn').textContent='Lecture auto';
        if (generatedText.trim()) chatHistory.push({role:'assistant', content: generatedText.trim()});
      }
    }
    function toggleAuto() {
      if(autoTimer){ clearInterval(autoTimer); autoTimer=null; $('auto-btn').textContent='Lecture auto'; return; }
      $('auto-btn').textContent='Pause'; autoTimer=setInterval(()=>{ if(currentIndex>=steps.length){ clearInterval(autoTimer); autoTimer=null; $('auto-btn').textContent='Lecture auto'; return; } showNextToken(); }, 260);
    }

    function updateFileLabel() {
      const names = selectedFiles.map(f => f.name);
      $('file-label').textContent = names.length ? names.join(', ') : 'Aucun fichier sélectionné';
      $('remove-file-btn').style.display = names.length ? 'inline-flex' : 'none';
    }
    function clearAttachments() {
      selectedFiles = [];
      $('file-input').value = '';
      updateFileLabel();
    }

    async function prepareGeneration() {
      resetGenerationPanel();
      const prompt=$('user-prompt').value.trim();
      if(!prompt){ $('user-prompt').focus(); return; }
      const useRag = $('use-rag').checked;
      const useNer = $('use-ner').checked;
      const useOllama = $('use-ollama').checked;
      const modeLabel = activeModeLabel(useRag, useNer, useOllama);
      openDrawer('generation');
      const fileNote = selectedFiles.length ? ` · ${selectedFiles.length} fichier(s)` : '';
      addMessage('user', prompt, 'Vous' + fileNote);
      chatHistory.push({role:'user', content: prompt});
      activeAssistantBubble = addMessage('assistant', 'Génération en cours...', modeLabel);

      const historyToSend = chatHistory.slice(0, -1).slice(-8);
      const form=new FormData();
      form.append('prompt',prompt);
      form.append('conversation_history', JSON.stringify(historyToSend));
      form.append('temperature',$('temperature').value);
      form.append('top_k',$('top-k').value);
      form.append('max_tokens',$('max-tokens').value);
      form.append('use_rag',useRag ? '1' : '0');
      form.append('use_ner',useNer ? '1' : '0');
      form.append('use_ollama',useOllama ? '1' : '0');
      for(const file of selectedFiles) form.append('files',file);
      try{
        const msg = `Analyse locale : ${modeLabel}...`;
        setLoading(true,msg); const res=await fetch('/api/prepare',{method:'POST',body:form}); const data=await res.json(); if(!res.ok||data.error) throw new Error(data.error||'Erreur de génération');
        steps=data.steps||[];
        if(useRag) renderRagResults(data.rag_chunks||[]); else $('rag-results').innerHTML='<p class="notice">RAG désactivé pour cette réponse.</p>';
        if(useNer) renderNerResults(data.ner_entities||[], data.ner_error||''); else $('ner-results').innerHTML='<p class="notice">NER désactivé pour cette réponse.</p>';
        currentIndex=0; generatedText='';
        if (activeAssistantBubble) activeAssistantBubble.textContent = '';
        $('next-btn').disabled=steps.length===0; $('auto-btn').disabled=steps.length===0; $('model-status').textContent=modeLabel; $('status-dot').className='dot ok';
        showTab('generation');
        if(steps.length) showNextToken(); else { if(activeAssistantBubble) activeAssistantBubble.textContent='Aucune réponse générée.'; }
        $('user-prompt').value='';
      } catch(err){
        $('model-status').textContent='Erreur'; $('status-dot').className='dot err';
        if(activeAssistantBubble) activeAssistantBubble.textContent='Erreur : '+err.message;
      }
      finally{ setLoading(false); }
    }
    async function checkModel() {
      try{ const res=await fetch('/api/status'); const data=await res.json(); if(data.loaded){ $('status-dot').className='dot ok'; $('model-status').textContent='NexusLM'; $('model-info').textContent=`Source : ${data.source} · ${data.vocab_path} · vocabulaire : ${data.vocab_size_file} mots · contexte : ${data.sequence_length} tokens`; } else { $('status-dot').className='dot err'; $('model-status').textContent='Erreur'; } }
      catch(err){ $('status-dot').className='dot err'; $('model-status').textContent='Erreur'; }
    }
    function escapeHtml(str){ return String(str).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;'); }
    $('file-input').addEventListener('change',()=>{ selectedFiles = Array.from($('file-input').files); updateFileLabel(); });
    $('composer-box').addEventListener('click',(event)=>{ if(event.target.id !== 'file-input') $('user-prompt').focus(); });
    $('user-prompt').addEventListener('keydown',(event)=>{
      if(event.key === 'Enter' && !event.shiftKey){
        event.preventDefault();
        prepareGeneration();
      }
    });
    setTheme(localStorage.getItem('nexus-theme') || 'light');
    syncControls(); resetMetrics(); checkModel();
    $('user-prompt').focus();
