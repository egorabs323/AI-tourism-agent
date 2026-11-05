const $ = id => document.getElementById(id);
let currentId = null;
let busy = false;
let uploading = false;
let authMode = 'login';
async function api(path, options={}) {
  const response = await fetch(path,{headers:{'Content-Type':'application/json'},...options});
  if (!response.ok) {
    let message = `Ошибка HTTP ${response.status}`;
    try { const data=await response.json(); message=data.detail || message; } catch(_e){}
    throw new Error(message);
  }
  return response.status===204 ? null : response.json();
}
function error(text='') {$('error').hidden=!text; $('error').textContent=text;}
function scrollBottom() {$('chat-scroll').scrollTop=$('chat-scroll').scrollHeight;}
function safeUrl(value) {
  try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; }
  catch (_) { return null; }
}
function buildSourceList(sources) {
  const unique = new Map();
  for (const s of sources || []) {
    const url = safeUrl(s.url);
    if (url && !unique.has(url)) unique.set(url, { ...s, url });
  }
  if (!unique.size) return null;
  const details=document.createElement('details'); details.className='source-list';
  const summary=document.createElement('summary');summary.textContent=`Источники (${unique.size})`;details.append(summary);
  for (const source of unique.values()) {
    const a=document.createElement('a');a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';
    const mark = Number.isInteger(Number(source.number)) ? `[${source.number}] ` : '';
    a.textContent=mark + (source.title || source.url); details.append(a);
  }
  return details;
}
function renderRoute(parsed) {
  const container = document.createElement('div'); container.className = 'route-view';
  if (parsed.preface) {
    const intro = document.createElement('p');intro.className='route-intro';intro.textContent=parsed.preface;container.append(intro);
  }
  for (const day of parsed.days) {
    const section=document.createElement('section');section.className='day-card';
    const heading=document.createElement('div');heading.className='day-heading';
    const title=document.createElement('h3');title.textContent=`День ${day.number}${day.title ? ' · ' + day.title : ''}`;
    const count=document.createElement('span');count.className='stop-count';count.textContent=`${day.stops.length} ${day.stops.length === 1 ? 'остановка' : day.stops.length < 5 ? 'остановки' : 'остановок'}`;
    heading.append(title,count);section.append(heading);
    if (day.message) {
      const message=document.createElement('p');message.className='stop-missing';message.textContent=day.message;section.append(message);
    }
    const list=document.createElement('ol');list.className='stops';
    for (const stop of day.stops) {
      const item=document.createElement('li');item.className='stop';
      const name=document.createElement('div');name.className='stop-title';name.textContent=stop.name;
      const line=document.createElement('div');line.className='stop-line';line.append(name);
      if (stop.source) {
        const ref=document.createElement('span');ref.className='source-ref';ref.textContent=`Источник ${stop.source}`;line.append(ref);
      }
      item.append(line);
      if (stop.description) {
        const description=document.createElement('p');description.className='stop-description';description.textContent=stop.description;
        item.append(description);
      } else {
        const missing=document.createElement('p');missing.className='stop-missing';missing.textContent='Описание в найденных материалах недостаточно подробное.';item.append(missing);
      }
      list.append(item);
    }
    section.append(list);container.append(section);
  }
  if (parsed.note) {
    const notice=document.createElement('p');notice.className='route-note';notice.textContent=parsed.note;container.append(notice);
  }
  return container;
}
function addMessage(role,text,sources=[]) {
  if ($('messages').querySelector('.empty')) $('messages').replaceChildren();
  const node=document.createElement('article');node.className=`message ${role}`;
  const who=document.createElement('div');who.className='who';who.textContent=role==='user'?'Вы':'Travel RAG';
  const bubble=document.createElement('div');bubble.className='bubble';
  if (role === 'assistant') {
    const parsed = TravelRouteFormat.parseAnswer(text);
    if (parsed.kind === 'route') {
      bubble.classList.add('route-bubble');bubble.append(renderRoute(parsed));
    } else { bubble.textContent = sources.length ? parsed.text : text; }
  } else { bubble.textContent = text; }
  node.append(who,bubble);
  if (sources.length && role==='assistant') {
    const list=buildSourceList(sources);if(list)node.append(list);
  }
  $('messages').append(node);scrollBottom();return node;
}
function showEmpty() { $('messages').innerHTML=`<section class="empty"><div class="empty-icon">⌁</div><h1>Куда отправимся?</h1><p>Выбери город и задай вопрос о путешествии.</p></section>`; }
async function refreshConversations(){
  const items=await api('/api/conversations');$('conversations').replaceChildren();
  for(const c of items){
    const row=document.createElement('div');row.className=`conversation ${c.id===currentId?'active':''}`;
    const select=document.createElement('span');select.className='conv-label';select.textContent=c.title;
    const del=document.createElement('button');del.className='delete';del.textContent='×';del.title='Удалить диалог';
    row.append(select,del);row.addEventListener('click',()=>openConversation(c.id));
    del.addEventListener('click',async e=>{e.stopPropagation();if(busy || uploading || !confirm('Удалить диалог?'))return;await api(`/api/conversations/${c.id}`,{method:'DELETE'});if(currentId===c.id){currentId=null;showEmpty()}await refreshConversations()});
    $('conversations').append(row);
  }
}
async function openConversation(id){
  if(busy || uploading)return;currentId=id;error();
  const records=await api(`/api/conversations/${id}/messages`);$('messages').replaceChildren();
  if(!records.length)showEmpty();else for(const item of records)addMessage(item.role,item.content,item.sources);
  await refreshConversations();await refreshDocuments();scrollBottom();
}
async function createConversation(){
  if(busy || uploading)return;
  const data=await api('/api/conversations',{method:'POST',body:JSON.stringify({})});
  await openConversation(data.id);$('question').focus();
}
async function send(){
  const question=$('question').value.trim();if(!question || busy)return;
  error();busy=true;$('send').disabled=true;$('send').textContent='Генерация…';
  try{
    if(!currentId){const data=await api('/api/conversations',{method:'POST',body:'{}'});currentId=data.id}
    addMessage('user',question);$('question').value='';
    const pending=addMessage('assistant','Ищу информацию в базе знаний…');
    try{
      const result=await api(`/api/conversations/${currentId}/messages`,{method:'POST',body:JSON.stringify({question,destination:$('destination').value.trim() || null,mode:$('request-mode').value,top_k:6})});
      pending.remove();addMessage('assistant',result.answer,result.sources);
      await refreshConversations();
    }catch(e){pending.remove();error(e.message);$('question').value=question;}
  }catch(e){error(e.message)}finally{busy=false;$('send').disabled=false;$('send').textContent='Отправить ↗';}
}

async function refreshDocuments(){
  const box=$('documents');box.replaceChildren();
  if(!currentId){$('upload-status').textContent='Создай диалог, чтобы добавить документы.';return;}
  const docs=await api(`/api/conversations/${currentId}/documents`);
  $('upload-status').textContent=docs.length ? `${docs.length} документ(ов) в этом диалоге` : 'Документы пока не загружены.';
  for(const doc of docs){
    const chip=document.createElement('div');chip.className='document-chip';
    const link=document.createElement('a');link.textContent=doc.filename;
    link.href=`/api/conversations/${currentId}/documents/${doc.id}/download`;
    link.title=`${doc.filename} · ${doc.chunks} чанков`;link.target='_blank';
    const remove=document.createElement('button');remove.type='button';remove.textContent='×';remove.title='Удалить документ';
    remove.addEventListener('click',async()=>{
      if(busy || uploading || !confirm(`Удалить ${doc.filename}?`))return;
      try{await api(`/api/conversations/${currentId}/documents/${doc.id}`,{method:'DELETE'});await refreshDocuments();}
      catch(e){error(e.message)}
    });
    chip.append(link,remove);box.append(chip);
  }
}
async function uploadDocument(){
  if(busy || uploading)return;
  const selected=$('file-input').files[0];
  if(!selected){error('Выбери PDF, DOCX или TXT.');return;}
  if(selected.size>5*1024*1024){error('Файл должен быть не больше 5 МБ.');return;}
  uploading=true;$('upload-btn').disabled=true;$('upload-status').textContent='Идёт обработка и индексация файла…';error();
  try{
    if(!currentId){const conv=await api('/api/conversations',{method:'POST',body:'{}'});currentId=conv.id;}
    const form=new FormData();form.append('file',selected);
    // При отправке FormData нельзя указывать Content-Type вручную.
    const response=await fetch(`/api/conversations/${currentId}/documents`,{method:'POST',body:form});
    if(!response.ok){let detail=`Ошибка HTTP ${response.status}`;try{detail=(await response.json()).detail || detail}catch(_){}throw new Error(detail)}
    const result=await response.json();$('file-input').value='';
    await refreshConversations();await refreshDocuments();
    $('upload-status').textContent=`Загружено: ${result.filename}, чанков: ${result.chunks}.`;
  }catch(e){error(e.message);$('upload-status').textContent='Не удалось загрузить документ.'}
  finally{uploading=false;$('upload-btn').disabled=false;}
}
async function checkAuth() {
  try {
    const me = await api('/api/auth/me');
    $('auth-screen').hidden = true;
    $('main-layout').hidden = false;
    $('account-name').textContent = me.username;
    currentId = null;
    await refreshConversations();
    await refreshDocuments();
  } catch(e) {
    $('main-layout').hidden = true;
    $('auth-screen').hidden = false;
  }
}
function selectAuthMode(mode) {
  authMode=mode;
  $('mode-login').classList.toggle('selected', mode==='login');
  $('mode-register').classList.toggle('selected', mode==='register');
  $('auth-submit').textContent=mode==='login'?'Войти':'Зарегистрироваться';
  $('auth-error').hidden=true;
  $('auth-password').setAttribute('autocomplete', mode==='login'?'current-password':'new-password');
}
$('mode-login').addEventListener('click',()=>selectAuthMode('login'));
$('mode-register').addEventListener('click',()=>selectAuthMode('register'));
$('auth-form').addEventListener('submit',async e=>{
  e.preventDefault();
  $('auth-error').hidden=true; $('auth-submit').disabled=true;
  try {
    await api(`/api/auth/${authMode}`, {method:'POST',body:JSON.stringify({username:$('auth-username').value,password:$('auth-password').value})});
    $('auth-password').value='';
    await checkAuth();
  } catch(err) {
    $('auth-error').hidden=false; $('auth-error').textContent=err.message;
  } finally { $('auth-submit').disabled=false; }
});
$('logout-btn').addEventListener('click',async()=>{
  if (busy || uploading) return;
  try {await api('/api/auth/logout',{method:'POST'});} catch(e) {error(e.message);return;}
  currentId=null; $('messages').replaceChildren(); showEmpty();
  $('conversations').replaceChildren(); $('documents').replaceChildren();
  $('main-layout').hidden=true; $('auth-screen').hidden=false;
});
$('upload-btn').addEventListener('click',uploadDocument);

$('chat-form').addEventListener('submit',e=>{e.preventDefault();send()});
$('question').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
$('new-chat').addEventListener('click',()=>createConversation().catch(e=>error(e.message)));
document.querySelectorAll('[data-prompt]').forEach(button=>button.addEventListener('click',()=>{const prompt=button.dataset.prompt;$('request-mode').value=/маршрут|за один день|за два дня/i.test(prompt)?'route':/музе/i.test(prompt)?'recommendations':'reference';$('request-mode').dispatchEvent(new Event('change'));$('question').value=prompt;send()}));
checkAuth().catch(e=>error(e.message));

// Подсказки зависят от выбранного режима; без новых запросов к серверу.
const modeExamples = {
  route:['Например: составь маршрут по Парижу на два дня', 'Маршрут по дням на основе Wikivoyage; точная навигация не рассчитывается.'],
  recommendations:['Например: посоветуй интересные музеи Парижа', 'Подбор достопримечательностей на основе источников.'],
  reference:['Например: что известно о Колизее?', 'Ответы на вопросы по туристическим статьям Wikivoyage.']
};
function updateModeHint(){
  const item=modeExamples[$('request-mode').value]||modeExamples.reference;
  $('question').placeholder=item[0];
  $('mode-hint').textContent=item[1];
}
$('request-mode').addEventListener('change', updateModeHint);
updateModeHint();
