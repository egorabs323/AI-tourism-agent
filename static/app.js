const $ = id => document.getElementById(id);
let currentId = null;
let busy = false;
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
    if (!day.stops.length) continue;
    const section=document.createElement('section');section.className='day-card';
    const heading=document.createElement('div');heading.className='day-heading';
    const title=document.createElement('h3');title.textContent=`День ${day.number}${day.title ? ' · ' + day.title : ''}`;
    const count=document.createElement('span');count.className='stop-count';count.textContent=`${day.stops.length} ${day.stops.length === 1 ? 'остановка' : day.stops.length < 5 ? 'остановки' : 'остановок'}`;
    heading.append(title,count);section.append(heading);
    const list=document.createElement('ol');list.className='stops';
    for (const stop of day.stops) {
      const item=document.createElement('li');item.className='stop';
      const name=document.createElement('div');name.className='stop-title';name.textContent=stop.name;
      const ref=document.createElement('span');ref.className='source-ref';ref.textContent=`Источник ${stop.source}`;
      const line=document.createElement('div');line.className='stop-line';line.append(name,ref);
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
    del.addEventListener('click',async e=>{e.stopPropagation();if(busy || !confirm('Удалить диалог?'))return;await api(`/api/conversations/${c.id}`,{method:'DELETE'});if(currentId===c.id){currentId=null;showEmpty()}await refreshConversations()});
    $('conversations').append(row);
  }
}
async function openConversation(id){
  if(busy)return;currentId=id;error();
  const records=await api(`/api/conversations/${id}/messages`);$('messages').replaceChildren();
  if(!records.length)showEmpty();else for(const item of records)addMessage(item.role,item.content,item.sources);
  await refreshConversations();scrollBottom();
}
async function createConversation(){
  if(busy)return;
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
      const result=await api(`/api/conversations/${currentId}/messages`,{method:'POST',body:JSON.stringify({question,destination:$('destination').value.trim() || null,top_k:6})});
      pending.remove();addMessage('assistant',result.answer,result.sources);
      await refreshConversations();
    }catch(e){pending.remove();error(e.message);$('question').value=question;}
  }catch(e){error(e.message)}finally{busy=false;$('send').disabled=false;$('send').textContent='Отправить ↗';}
}
$('chat-form').addEventListener('submit',e=>{e.preventDefault();send()});
$('question').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
$('new-chat').addEventListener('click',()=>createConversation().catch(e=>error(e.message)));
document.querySelectorAll('[data-prompt]').forEach(button=>button.addEventListener('click',()=>{$('question').value=button.dataset.prompt;send()}));
refreshConversations().catch(e=>error(e.message));
