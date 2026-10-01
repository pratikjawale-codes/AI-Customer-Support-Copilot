// Virtual Assistant Customer Support frontend

let activeTicket = null;
let currentStyle = 'professional';
let currentResponses = {};
let currentContextDocs = [];
let obsChartInstance = null;
let ticketQueue = [];
let analysisRun = 0;

function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[character]);
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  checkEngineHealth();
  loadTickets();
  loadKnowledgeBaseDocs();
  initObservabilityChart();
});

// Tab Switcher
function switchTab(tabId, button) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active', 'text-white'));
  
  const selectedTab = document.getElementById(`tab-${tabId}`);
  if (selectedTab) selectedTab.classList.remove('hidden');

  if (button) button.classList.add('active', 'text-white');

  if (tabId === 'observability') {
    fetchObservabilityStats();
  }
}

// Check Health & Engine Mode
async function checkEngineHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const badge = document.getElementById('engine-status-badge');
    const text = document.getElementById('engine-mode-text');

    if (data.mode === 'LIVE_OPENAI') {
      badge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-full bg-emerald-950 border border-emerald-800 text-xs text-emerald-300";
      text.innerText = "Live OpenAI Mode (API Keys Active)";
    } else {
      badge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-full bg-amber-950 border border-amber-800 text-xs text-amber-300";
      text.innerText = "Simulated Local Mode (API Keys Empty)";
    }
  } catch (err) {
    console.error("Health check failed", err);
  }
}

// Load Support Tickets List
async function loadTickets() {
  try {
    const res = await fetch('/api/mock-tickets');
    if (!res.ok) throw new Error(`Ticket request failed (${res.status})`);
    const tickets = await res.json();
    ticketQueue = tickets;
    renderTicketQueue();
  } catch (err) {
    console.error("Failed loading tickets", err);
  }
}

function renderTicketQueue() {
    const container = document.getElementById('tickets-list');
    container.innerHTML = '';

    ticketQueue.forEach((t, index) => {
      const card = document.createElement('div');
      card.className = `p-3 rounded-xl border transition cursor-pointer space-y-1.5 ${index === 0 ? 'bg-slate-800/90 border-blue-500/80 shadow-md' : 'bg-slate-900/60 border-slate-800 hover:bg-slate-800/50'}`;
      card.onclick = (e) => selectTicket(t, card);

      const priorityColor = t.priority === 'High' ? 'text-rose-400 bg-rose-950/60 border-rose-800' : 'text-amber-400 bg-amber-950/60 border-amber-800';

      card.innerHTML = `
        <div class="flex items-center justify-between text-xs">
          <span class="font-mono text-blue-400 font-semibold">${escapeHTML(t.id)}</span>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold border ${priorityColor}">${escapeHTML(t.priority)}</span>
        </div>
        <h4 class="text-xs font-bold text-slate-200">${escapeHTML(t.customer_name)}</h4>
        <p class="text-xs text-slate-400 line-clamp-2">${escapeHTML(t.ticket_text)}</p>
      `;

      container.appendChild(card);
    });

    document.getElementById('ticket-count').innerText = `${ticketQueue.length} Tickets`;
    if (ticketQueue.length > 0) {
      selectTicket(ticketQueue[0], container.children[0]);
    }
}

// Select Ticket & Execute Copilot Workbench
async function selectTicket(ticket, cardElement) {
  activeTicket = ticket;
  
  // Highlight active ticket card
  document.querySelectorAll('#tickets-list > div').forEach(c => {
    c.className = 'p-3 rounded-xl border transition cursor-pointer space-y-1.5 bg-slate-900/60 border-slate-800 hover:bg-slate-800/50';
  });
  if (cardElement) {
    cardElement.className = 'p-3 rounded-xl border transition cursor-pointer space-y-1.5 bg-slate-800/90 border-blue-500/80 shadow-md';
  }

  // Populate UI
  document.getElementById('active-ticket-id').innerText = ticket.id;
  document.getElementById('active-customer-name').innerText = ticket.customer_name;
  document.getElementById('active-ticket-text').innerText = `"${ticket.ticket_text}"`;
  document.getElementById('hitl-status-message').classList.add('hidden');

  // Trigger Classification & Sentiment APIs concurrently
  runCopilotAnalysis(ticket.ticket_text);
}

// Run Copilot Classification, RAG & Generation
async function runCopilotAnalysis(ticketText) {
  const run = ++analysisRun;
  try {
    // 1. Classification & Sentiment
    const [classRes, sentRes] = await Promise.all([
      fetch('/api/tickets/classify', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ticket_text: ticketText})
      }).then(r => r.json()),
      fetch('/api/tickets/sentiment', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ticket_text: ticketText})
      }).then(r => r.json())
    ]);
    if (run !== analysisRun) return;

    // Update Badges
    document.getElementById('badge-category').innerText = classRes.category;
    document.getElementById('badge-intent').innerText = classRes.subcategory;
    document.getElementById('badge-priority').innerText = classRes.priority;
    document.getElementById('badge-sentiment').innerText = `${sentRes.sentiment} (${sentRes.confidence})`;

    const seniorBadge = document.getElementById('active-senior-route');
    if (sentRes.route_to_senior) {
      seniorBadge.classList.remove('hidden');
    } else {
      seniorBadge.classList.add('hidden');
    }

    // 2. RAG Retrieval
    const ragRes = await fetch('/api/rag/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: ticketText, top_k: 2})
    }).then(r => r.json());
    if (run !== analysisRun) return;

    currentContextDocs = ragRes.retrieved_documents.map(d => d.content);
    renderRAGCards(ragRes.retrieved_documents);

    // 3. Response Generation
    const genRes = await fetch('/api/response/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        ticket_text: ticketText,
        context_docs: currentContextDocs,
        style: currentStyle
      })
    }).then(r => r.json());
    if (run !== analysisRun) return;

    currentResponses = genRes;
    document.getElementById('hitl-response-editor').value = genRes[currentStyle];

  } catch (err) {
    console.error("Copilot analysis error", err);
  }
}

// Render RAG Cards
function renderRAGCards(documents) {
  const container = document.getElementById('rag-context-container');
  container.innerHTML = '';

  if (!documents || documents.length === 0) {
    container.innerHTML = `<div class="col-span-2 p-3 bg-slate-900 rounded-lg text-xs text-slate-500">No matching KB policy found.</div>`;
    return;
  }

  documents.forEach(doc => {
    const card = document.createElement('div');
    card.className = "p-3 bg-slate-900/90 rounded-xl border border-slate-800 space-y-1";
    card.innerHTML = `
      <div class="flex items-center justify-between text-xs">
        <span class="font-bold text-blue-300">${escapeHTML(doc.title)}</span>
        <span class="px-2 py-0.5 rounded text-[10px] bg-indigo-950 text-indigo-300 font-mono">Score: ${escapeHTML(doc.score)}</span>
      </div>
      <p class="text-xs text-slate-400 line-clamp-3">${escapeHTML(doc.content)}</p>
    `;
    container.appendChild(card);
  });
}

// Response Style Toggle
function setResponseStyle(style) {
  currentStyle = style;
  ['professional', 'short', 'empathetic'].forEach(s => {
    const btn = document.getElementById(`style-btn-${s}`);
    if (s === style) {
      btn.className = "px-3 py-1 rounded-md font-medium text-slate-200 bg-slate-800 shadow";
    } else {
      btn.className = "px-3 py-1 rounded-md font-medium text-slate-400 hover:text-slate-200";
    }
  });

  if (currentResponses[style]) {
    document.getElementById('hitl-response-editor').value = currentResponses[style];
  }
}

// Submit HITL Approval
async function submitHITL(action) {
  const editorText = document.getElementById('hitl-response-editor').value;
  try {
    const res = await fetch('/api/hitl/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        ticket_id: activeTicket ? activeTicket.id : 'TCK-1001',
        customer_query: activeTicket ? activeTicket.ticket_text : '',
        ai_suggested_response: currentResponses[currentStyle] || '',
        edited_response: editorText,
        action: action,
        agent_id: 'agent_007'
      })
    }).then(r => r.json());

    const statusBox = document.getElementById('hitl-status-message');
    const statusText = document.getElementById('hitl-status-text');
    statusText.innerText = res.message;
    statusBox.classList.remove('hidden');

    // Also record feedback
    fetch('/api/feedback', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        ticket_id: activeTicket ? activeTicket.id : 'TCK-1001',
        question: activeTicket ? activeTicket.ticket_text : '',
        retrieved_context: currentContextDocs,
        ai_response: currentResponses[currentStyle] || '',
        agent_edited_response: editorText,
        rating: action === 'approved' ? 'helpful' : 'not_helpful'
      })
    });

  } catch (err) {
    console.error("HITL submission error", err);
  }
}

// Regenerate Response
function regenerateResponse() {
  if (activeTicket) {
    runCopilotAnalysis(activeTicket.ticket_text);
  }
}

// Execute Agentic Workflow
async function runAgenticWorkflow() {
  const query = document.getElementById('agent-query-input').value;
  const container = document.getElementById('agent-steps-container');
  container.innerHTML = `<div class="p-6 text-center text-xs text-indigo-400 shimmer rounded-xl">Evaluating Decision Graph and executing tool calls...</div>`;

  try {
    const res = await fetch('/api/agent/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_query: query})
    }).then(r => r.json());

    container.innerHTML = '';

    // Header Node
    const intentNode = document.createElement('div');
    intentNode.className = "p-4 bg-indigo-950/80 border border-indigo-800 rounded-xl flex items-center justify-between text-xs text-indigo-200 font-bold mb-4";
    intentNode.innerHTML = `
      <span>Detected Intent: ${escapeHTML(res.detected_intent)}</span>
      <span class="font-mono text-purple-300">Tool Calls Invoked: ${res.execution_steps.length}</span>
    `;
    container.appendChild(intentNode);

    // Step Nodes
    res.execution_steps.forEach((step, idx) => {
      const card = document.createElement('div');
      card.className = "p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-2 relative";
      card.innerHTML = `
        <div class="flex items-center justify-between text-xs">
          <span class="px-2 py-0.5 rounded bg-blue-950 text-blue-300 font-mono font-bold">Step ${escapeHTML(step.step_number)}: ${escapeHTML(step.tool_name)}()</span>
          <span class="text-slate-500 font-mono">${escapeHTML(step.latency_ms)} ms</span>
        </div>
        <p class="text-xs text-slate-300 italic">" ${escapeHTML(step.thought)} "</p>
        <div class="p-2.5 bg-slate-950 rounded border border-slate-800/80 text-xs font-mono text-emerald-400 overflow-x-auto">
          <span class="text-slate-500 block text-[10px]">Tool Output:</span>
          ${escapeHTML(JSON.stringify(step.tool_output, null, 2))}
        </div>
      `;
      container.appendChild(card);
    });

    // Final Response Node
    const finalCard = document.createElement('div');
    finalCard.className = "p-4 bg-slate-800/90 border border-emerald-600/80 rounded-xl space-y-2 mt-4";
    finalCard.innerHTML = `
      <div class="flex items-center justify-between text-xs font-bold text-emerald-400">
        <span class="flex items-center space-x-1.5"><i data-lucide="check-circle" class="w-4 h-4"></i> Final Response Synthesis</span>
        <span class="text-slate-400 text-[10px]">Requires Human Approval</span>
      </div>
      <p class="text-xs text-slate-100 font-sans">${escapeHTML(res.final_response)}</p>
    `;
    container.appendChild(finalCard);
    lucide.createIcons();

  } catch (err) {
    container.innerHTML = `<div class="p-4 bg-rose-950 text-rose-300 rounded-xl text-xs">Agent execution failed.</div>`;
  }
}

// Guardrail Inspector
async function testGuardrails() {
  const text = document.getElementById('guardrail-input').value;
  try {
    const res = await fetch('/api/guardrails/check', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text})
    }).then(r => r.json());

    const resultBox = document.getElementById('guardrail-result');
    const badge = document.getElementById('guard-status-badge');
    const reason = document.getElementById('guard-reason');
    const sanitized = document.getElementById('guard-sanitized');

    resultBox.classList.remove('hidden');

    if (!res.is_safe) {
      badge.className = "px-2.5 py-1 rounded text-xs font-bold bg-rose-950 text-rose-300 border border-rose-800";
      badge.innerText = `BLOCKED: ${res.flags.join(', ')}`;
      reason.innerText = res.block_reason || "Violation detected.";
      sanitized.innerText = res.sanitized_text;
    } else {
      badge.className = "px-2.5 py-1 rounded text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-800";
      badge.innerText = "PASSED: Input & Output Safe";
      reason.innerText = "No prompt injection or unauthorized financial claims detected.";
      sanitized.innerText = res.sanitized_text;
    }
  } catch (err) {
    console.error("Guardrail check error", err);
  }
}

// PII Protection
async function testPIIMasking() {
  const text = document.getElementById('pii-input').value;
  try {
    const res = await fetch('/api/pii/mask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text})
    }).then(r => r.json());

    const resultBox = document.getElementById('pii-result');
    const typesContainer = document.getElementById('pii-types-container');
    const output = document.getElementById('pii-masked-output');

    resultBox.classList.remove('hidden');
    typesContainer.innerHTML = '';

    res.detected_types.forEach(t => {
      const chip = document.createElement('span');
      chip.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800";
      chip.innerText = t;
      typesContainer.appendChild(chip);
    });

    output.innerText = res.masked_text;
  } catch (err) {
    console.error("PII error", err);
  }
}

// Summarizer
async function generateSummary() {
  const text = document.getElementById('summary-transcript-input').value;
  const lines = text.split('\n').filter(l => l.trim().length > 0);
  const conversation = lines.map(l => {
    const parts = l.split(':');
    return {
      sender: parts[0] ? parts[0].trim() : 'User',
      text: parts.slice(1).join(':').trim()
    };
  });

  try {
    const res = await fetch('/api/summarize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({conversation: conversation})
    }).then(r => r.json());

    document.getElementById('sum-issue').innerText = res.issue;
    document.getElementById('sum-actions').innerText = res.actions_taken;
    document.getElementById('sum-status').innerText = res.current_status;
    document.getElementById('sum-next').innerText = res.next_action;
  } catch (err) {
    console.error("Summary error", err);
  }
}

// Observability Stats
async function fetchObservabilityStats() {
  try {
    const res = await fetch('/api/observability/stats').then(r => r.json());
    document.getElementById('obs-total-req').innerText = res.total_requests;
    document.getElementById('obs-avg-lat').innerText = `${res.avg_latency_ms} ms`;
    document.getElementById('obs-tokens').innerText = res.total_tokens_used.toLocaleString();
    document.getElementById('obs-cost').innerText = `$${res.total_estimated_cost_usd}`;
    const logs = res.recent_request_logs || [];
    if (obsChartInstance) {
      obsChartInstance.data.labels = logs.map(log => log.timestamp);
      obsChartInstance.data.datasets[0].data = logs.map(log => log.latency_ms);
      obsChartInstance.data.datasets[1].data = logs.map(log => log.total_tokens);
      obsChartInstance.update();
    }

    const table = document.getElementById('obs-logs-table');
    table.innerHTML = '';
    logs.forEach(log => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="p-2 text-blue-400 font-bold">${escapeHTML(log.request_id)}</td>
        <td class="p-2 text-slate-300">${escapeHTML(log.endpoint)}</td>
        <td class="p-2 text-slate-400">${escapeHTML(log.latency_ms)} ms</td>
        <td class="p-2 text-purple-300">${escapeHTML(log.total_tokens)}</td>
        <td class="p-2 text-emerald-400">$${escapeHTML(log.cost_usd)}</td>
      `;
      table.appendChild(tr);
    });
  } catch (err) {
    console.error("Observability error", err);
  }
}

// Chart.js Setup
function initObservabilityChart() {
  const ctx = document.getElementById('observabilityChart');
  if (!ctx) return;

  obsChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Latency (ms)',
          data: [],
          borderColor: '#3b82f6',
          tension: 0.3
        },
        {
          label: 'Tokens',
          data: [],
          borderColor: '#a855f7',
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#94a3b8' } }
      },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
        y: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } }
      }
    }
  });
}

// Knowledge Base Loader
async function loadKnowledgeBaseDocs() {
  const response = await fetch('/api/knowledge-base');
  if (!response.ok) throw new Error(`Knowledge base request failed (${response.status})`);
  const docs = await response.json();
  const grid = document.getElementById('kb-documents-grid');
  if (!grid) return;
  grid.innerHTML = '';

  docs.forEach(d => {
    const card = document.createElement('div');
    card.className = "p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-2";
    card.innerHTML = `
      <div class="flex items-center justify-between">
        <span class="font-bold text-xs text-slate-200">${escapeHTML(d.title)}</span>
        <span class="px-2 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-300 font-bold border border-emerald-800">${escapeHTML(d.status)}</span>
      </div>
      <div class="flex items-center justify-between text-xs text-slate-400">
        <span>Category: ${escapeHTML(d.category)}</span>
        <span class="font-mono text-indigo-400">${escapeHTML(d.chunks)} Chunks</span>
      </div>
    `;
    grid.appendChild(card);
  });
}

async function refreshKB() {
  try {
    const response = await fetch('/api/knowledge-base/reindex', {method: 'POST'});
    if (!response.ok) throw new Error(`Re-index failed (${response.status})`);
    await loadKnowledgeBaseDocs();
  } catch (err) {
    console.error("Knowledge base re-index failed", err);
  }
}

function createNewTicketModal() {
  const ticketText = window.prompt('Describe the customer issue:');
  if (!ticketText || !ticketText.trim()) return;
  const customerName = window.prompt('Customer name:', 'New Customer') || 'New Customer';
  const ticket = {
    id: `TCK-${Date.now().toString().slice(-6)}`,
    customer_name: customerName.trim(),
    ticket_text: ticketText.trim(),
    priority: 'Medium'
  };
  ticketQueue.unshift(ticket);
  renderTicketQueue();
}
