/**
 * Nalanda AI Frontend Controller
 *
 * CRITICAL CONSTRAINT:
 * Extracted text, OCR output, raw Markdown, and JSON extraction results
 * are kept strictly internal to the backend. This frontend displays
 * ONLY document-level metadata and processing status.
 */

// State
let token = localStorage.getItem("nalanda_token");
let currentUser = null;
let folders = [];
let activeFolderId = null;
let documents = [];
let pollingTimer = null;
let isRegisterMode = false;
let searchController = null;

// Helpers
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 4000);
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function formatDate(isoString) {
  if (!isoString) return "-";
  try {
    const d = new Date(isoString);
    return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return isoString;
  }
}

async function apiFetch(url, options = {}) {
  const headers = { ...options.headers };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  const response = await fetch(url, { ...options, headers });
  if (response.status === 401) {
    // Session expired or invalid
    localStorage.removeItem("nalanda_token");
    token = null;
    updateAuthUI();
    openAuthModal();
    throw new Error("Session expired. Please sign in again.");
  }
  return response;
}

// Authentication
async function checkAuth() {
  if (!token) {
    // If no token, log in with demo account automatically for convenience
    await handleDemoLogin();
    return;
  }
  try {
    const res = await apiFetch("/api/auth/me");
    if (res.ok) {
      currentUser = await res.json();
      updateAuthUI();
      await loadFolders();
    } else {
      await handleDemoLogin();
    }
  } catch (err) {
    console.error("Auth check failed:", err);
  }
}

function updateAuthUI() {
  const badge = document.getElementById("user-profile-badge");
  const avatar = document.getElementById("user-avatar-text");
  const name = document.getElementById("user-display-name");
  const btnLogout = document.getElementById("btn-logout");

  if (currentUser) {
    badge.style.display = "flex";
    avatar.textContent = (currentUser.username || "U")[0].toUpperCase();
    name.textContent = currentUser.username;
    btnLogout.style.display = "inline-flex";
  } else {
    badge.style.display = "none";
    btnLogout.style.display = "none";
  }
}

async function handleDemoLogin() {
  try {
    // Try to register demo user or log in if already exists
    let res = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: "demo@nalanda.ai",
        username: "demo_user",
        password: "demopassword123",
      }),
    });

    if (!res.ok) {
      // User likely exists, log in
      res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email_or_username: "demo@nalanda.ai",
          password: "demopassword123",
        }),
      });
    }

    if (res.ok) {
      const data = await res.json();
      token = data.access_token;
      currentUser = data.user;
      localStorage.setItem("nalanda_token", token);
      closeAuthModal();
      updateAuthUI();
      showToast(`Signed in as ${currentUser.username}`);
      await loadFolders();
    }
  } catch (err) {
    showToast("Failed to authenticate demo user", "error");
  }
}

function openAuthModal() {
  document.getElementById("auth-modal").classList.add("active");
}

function closeAuthModal() {
  document.getElementById("auth-modal").classList.remove("active");
}

function toggleAuthMode(event) {
  event.preventDefault();
  isRegisterMode = !isRegisterMode;
  const title = document.getElementById("auth-modal-title");
  const btn = document.getElementById("btn-auth-submit");
  const toggleLink = document.getElementById("auth-toggle-link");
  const usernameGroup = document.getElementById("username-group");

  if (isRegisterMode) {
    title.textContent = "Create an Account";
    btn.textContent = "Register";
    toggleLink.textContent = "Already have an account? Sign in";
    usernameGroup.style.display = "block";
  } else {
    title.textContent = "Sign In to Nalanda AI";
    btn.textContent = "Sign In";
    toggleLink.textContent = "Need an account? Register here";
    usernameGroup.style.display = "none";
  }
}

async function handleAuthSubmit() {
  const emailOrUser = document.getElementById("auth-email").value.trim();
  const password = document.getElementById("auth-password").value;
  const username = document.getElementById("auth-username").value.trim();

  if (!emailOrUser || !password) {
    showToast("Please fill in all required fields", "error");
    return;
  }

  try {
    let res;
    if (isRegisterMode) {
      if (!username) {
        showToast("Please enter a username", "error");
        return;
      }
      res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: emailOrUser, username, password }),
      });
    } else {
      res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email_or_username: emailOrUser, password }),
      });
    }

    const data = await res.json();
    if (!res.ok) {
      showToast(data.detail || "Authentication failed", "error");
      return;
    }

    token = data.access_token;
    currentUser = data.user;
    localStorage.setItem("nalanda_token", token);
    closeAuthModal();
    updateAuthUI();
    showToast(`Welcome, ${currentUser.username}!`);
    await loadFolders();
  } catch (err) {
    showToast("An error occurred during authentication", "error");
  }
}

async function logout() {
  try {
    await apiFetch("/api/auth/logout", { method: "POST" });
  } catch (err) {
    // Ignore error
  }
  localStorage.removeItem("nalanda_token");
  token = null;
  currentUser = null;
  folders = [];
  activeFolderId = null;
  documents = [];
  if (pollingTimer) clearInterval(pollingTimer);
  updateAuthUI();
  renderFolders();
  renderActiveView();
  showToast("Logged out successfully.");
}

// Folders Management
async function loadFolders() {
  try {
    const res = await apiFetch("/api/folders");
    if (res.ok) {
      folders = await res.json();
      renderFolders();

      // If no folder selected yet and folders exist, select first
      if (!activeFolderId && folders.length > 0) {
        selectFolder(folders[0].id);
      } else if (activeFolderId) {
        // Refresh active folder
        selectFolder(activeFolderId);
      } else {
        renderActiveView();
      }
    }
  } catch (err) {
    console.error("Failed to load folders:", err);
  }
}

function renderFolders() {
  const container = document.getElementById("folder-list-container");
  container.innerHTML = "";

  if (folders.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="padding: 24px 12px;">
        <p style="font-size: 0.85rem; color: var(--text-muted);">No folders yet. Click '+ New Folder' to create one.</p>
      </div>
    `;
    return;
  }

  folders.forEach((folder) => {
    const item = document.createElement("div");
    item.className = `folder-item ${folder.id === activeFolderId ? "active" : ""}`;
    item.onclick = () => selectFolder(folder.id);

    item.innerHTML = `
      <div class="folder-info">
        <span class="folder-icon">📁</span>
        <span class="folder-name" title="${folder.name}">${folder.name}</span>
      </div>
      <span class="doc-count-badge">${folder.document_count || 0}</span>
    `;
    container.appendChild(item);
  });
}

function openCreateFolderModal() {
  document.getElementById("new-folder-name").value = "";
  document.getElementById("create-folder-modal").classList.add("active");
  setTimeout(() => document.getElementById("new-folder-name").focus(), 100);
}

function closeCreateFolderModal() {
  document.getElementById("create-folder-modal").classList.remove("active");
}

async function handleCreateFolder() {
  const nameInput = document.getElementById("new-folder-name");
  const name = nameInput.value.trim();
  if (!name) {
    showToast("Please enter a folder name", "error");
    return;
  }

  try {
    const res = await apiFetch("/api/folders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });

    if (res.ok) {
      const newFolder = await res.json();
      closeCreateFolderModal();
      showToast(`Created folder "${newFolder.name}"`);
      await loadFolders();
      selectFolder(newFolder.id);
    } else {
      const data = await res.json();
      showToast(data.detail || "Failed to create folder", "error");
    }
  } catch (err) {
    showToast("Error creating folder", "error");
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const folderChatHistory = {};
let activeConversationId = null;
let folderConversations = [];

async function loadFolderConversations(folderId) {
  const selectEl = document.getElementById("chat-conversation-select");

  try {
    const res = await apiFetch(`/api/folders/${folderId}/conversations`);
    if (!res.ok) throw new Error("Failed to load conversations");
    folderConversations = await res.json();

    if (selectEl) {
      selectEl.innerHTML = "";
      if (folderConversations.length > 0) {
        selectEl.style.display = "inline-block";
        folderConversations.forEach(c => {
          const opt = document.createElement("option");
          opt.value = c.id;
          opt.textContent = `${c.title} (${c.message_count} msgs)`;
          selectEl.appendChild(opt);
        });

        // Default to the first (most recent) conversation
        activeConversationId = folderConversations[0].id;
        selectEl.value = activeConversationId;
        await loadConversationMessages(activeConversationId);
      } else {
        selectEl.style.display = "none";
        activeConversationId = null;
        renderChatWelcome();
      }
    }
  } catch (err) {
    console.warn("Could not load folder conversations:", err);
    if (selectEl) selectEl.style.display = "none";
    activeConversationId = null;
    renderChatWelcome();
  }
}

function renderChatWelcome() {
  const container = document.getElementById("chat-messages");
  if (!container) return;
  container.innerHTML = `
    <div class="chat-message assistant">
      <div class="chat-avatar">AI</div>
      <div class="chat-bubble">
        <p>Hello! Ask me any question about the documents in this folder. You can also request diagrams (e.g. <em>"Show diagram of CNN"</em>), comparison tables, or full subtopic breakdowns.</p>
      </div>
    </div>
  `;
}

async function loadConversationMessages(conversationId) {
  const container = document.getElementById("chat-messages");
  if (!container || !conversationId) return;

  try {
    const res = await apiFetch(`/api/conversations/${conversationId}`);
    if (!res.ok) throw new Error("Failed to load conversation messages");
    const data = await res.json();

    container.innerHTML = "";
    if (!data.messages || data.messages.length === 0) {
      renderChatWelcome();
      return;
    }

    data.messages.forEach(msg => {
      const msgEl = document.createElement("div");
      msgEl.className = `chat-message ${msg.role === 'user' ? 'user' : 'assistant'}`;

      if (msg.role === 'user') {
        msgEl.innerHTML = `
          <div class="chat-avatar">You</div>
          <div class="chat-bubble">
            <p>${escapeHtml(msg.content)}</p>
          </div>
        `;
      } else {
        let bubbleContent = "";

        if (msg.direct_answer) {
          const directAnswerElId = `direct-answer-${msg.id}`;
          const isRefined = msg.refinement_metadata && msg.refinement_metadata.is_refined;
          const badge = isRefined
            ? `<span style="display:inline-flex; align-items:center; gap:4px; font-size:0.75rem; background:rgba(99, 102, 241, 0.15); color:#818cf8; border:1px solid rgba(99, 102, 241, 0.3); padding:2px 8px; border-radius:12px; margin-left:8px; font-weight:500;">✨ Refined</span>`
            : '';
          bubbleContent += `
            <div class="direct-answer-card">
              <div class="direct-answer-title" style="display:flex; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center;">
                  <span>💡 Direct Answer</span>
                  ${badge}
                </div>
              </div>
              <div class="direct-answer-body" id="${directAnswerElId}">${formatDirectAnswerHtml(msg.direct_answer)}</div>
            </div>
          `;
        } else {
          bubbleContent += `<div class="direct-answer-body">${formatDirectAnswerHtml(msg.content)}</div>`;
        }

        if (msg.primary_figure) {
          const fig = msg.primary_figure;
          const figUrl = token ? `${fig.figure_url}?token=${encodeURIComponent(token)}` : fig.figure_url;
          const encodedCap = encodeURIComponent(fig.caption || "");
          bubbleContent += `
            <div class="diagram-card">
              <div class="diagram-header">
                <span>🖼️ Extracted Diagram (Page ${fig.page_number || '1'})</span>
                <span style="font-size:0.75rem; color:var(--text-muted); cursor:pointer;" onclick="openFigureModal('${figUrl}', '${encodedCap}')">Click to zoom</span>
              </div>
              <img class="diagram-preview-img" src="${figUrl}" alt="Diagram" onclick="openFigureModal('${figUrl}', '${encodedCap}')">
              ${fig.caption ? `<div class="diagram-caption">${escapeHtml(fig.caption)}</div>` : ''}
            </div>
          `;
        }

        if (msg.primary_table && msg.primary_table.html) {
          const tab = msg.primary_table;
          bubbleContent += `
            <div class="table-card">
              <div class="diagram-header" style="margin-bottom:6px;">
                <span>📊 Extracted Table (Page ${tab.page_number || '1'})</span>
              </div>
              ${tab.html}
            </div>
          `;
        }

        if (msg.citations && msg.citations.length > 0) {
          bubbleContent += `
            <div class="citations-group">
              <span class="citations-label">Sources:</span>
              ${msg.citations.map(r => `
                <span class="citation-pill" title="${r.section_title ? escapeHtml(r.section_title) : ''}">
                  📄 <strong>${escapeHtml(r.source_filename)}</strong>
                  ${r.page_number != null ? `· p. ${r.page_number}` : ''}
                  ${r.section_title ? `· ${escapeHtml(r.section_title.substring(0, 30))}` : ''}
                  <span style="color:var(--status-completed-text); font-weight:600;">(${Math.round(r.relevance_score * 100)}%)</span>
                </span>
              `).join('')}
            </div>
          `;
        }

        msgEl.innerHTML = `
          <div class="chat-avatar">AI</div>
          <div class="chat-bubble">
            ${bubbleContent}
          </div>
        `;
      }

      container.appendChild(msgEl);
    });

    container.scrollTop = container.scrollHeight;
  } catch (err) {
    console.error("Error loading messages:", err);
  }
}

async function switchConversation(conversationId) {
  if (!conversationId) return;
  activeConversationId = conversationId;
  await loadConversationMessages(conversationId);
}

function startNewChat() {
  activeConversationId = null;
  const selectEl = document.getElementById("chat-conversation-select");
  if (selectEl) selectEl.selectedIndex = -1;
  renderChatWelcome();
  const queryInput = document.getElementById("rag-query");
  if (queryInput) queryInput.focus();
}

async function clearChatHistory() {
  if (!activeConversationId) {
    renderChatWelcome();
    return;
  }
  try {
    await apiFetch(`/api/conversations/${activeConversationId}`, { method: "DELETE" });
    if (activeFolderId) {
      await loadFolderConversations(activeFolderId);
    }
  } catch (err) {
    console.error("Error deleting conversation:", err);
    renderChatWelcome();
  }
}

function openFigureModal(url, encodedCaption) {
  const modal = document.getElementById("figure-modal");
  const img = document.getElementById("figure-modal-img");
  const cap = document.getElementById("figure-modal-caption");
  if (modal && img) {
    img.src = url;
    if (cap) cap.textContent = decodeURIComponent(encodedCaption || "");
    modal.classList.add("active");
  }
}

function closeFigureModal() {
  const modal = document.getElementById("figure-modal");
  if (modal) modal.classList.remove("active");
}

async function selectFolder(folderId) {
  if (searchController) {
    searchController.abort();
    searchController = null;
    const submitBtn = document.getElementById("rag-submit");
    if (submitBtn) submitBtn.disabled = false;
  }
  activeFolderId = folderId;
  renderFolders();
  renderActiveView();
  await loadFolderConversations(folderId);
  await loadDocuments(folderId);
}

function formatInlineMarkdown(str) {
  let esc = escapeHtml(str);
  esc = esc.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  esc = esc.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08); padding:1px 5px; border-radius:4px; font-family:monospace; font-size:0.9em;">$1</code>');
  return esc;
}

function formatDirectAnswerHtml(text) {
  if (!text) return "";
  const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
  const htmlParts = [];
  let inList = false;

  for (const line of lines) {
    const headingMatch = line.match(/^#{1,4}\s+(.+)$/);
    const stepMatch = line.match(/^(\d+[\.\)])\s*(.+)$/);
    const bulletMatch = line.match(/^(?:[-•+]|\*(?!\*))\s*(.+)$/);

    if (headingMatch) {
      if (inList) {
        htmlParts.push("</ul>");
        inList = false;
      }
      htmlParts.push(`<div class="direct-answer-section-header" style="font-weight:700; margin-top:8px; margin-bottom:4px; color:var(--text-primary);">${formatInlineMarkdown(headingMatch[1])}</div>`);
    } else if (line.includes(" → ")) {
      if (inList) {
        htmlParts.push("</ul>");
        inList = false;
      }
      htmlParts.push(`<div class="direct-answer-flow" style="background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25); border-radius:8px; padding:10px 14px; margin:10px 0; font-size:0.95rem; line-height:1.6; color:var(--text-primary);">${formatInlineMarkdown(line)}</div>`);
    } else if (stepMatch) {
      if (inList) {
        htmlParts.push("</ul>");
        inList = false;
      }
      const num = stepMatch[1];
      const rest = stepMatch[2];
      htmlParts.push(`
        <div class="direct-answer-step">
          <span class="direct-answer-step-num">${escapeHtml(num)}</span>
          <span class="direct-answer-step-text">${formatInlineMarkdown(rest)}</span>
        </div>
      `);
    } else if (bulletMatch) {
      if (!inList) {
        htmlParts.push('<ul class="direct-answer-bullet-list">');
        inList = true;
      }
      htmlParts.push(`<li>${formatInlineMarkdown(bulletMatch[1])}</li>`);
    } else {
      if (inList) {
        htmlParts.push("</ul>");
        inList = false;
      }
      if (line.length < 80 && !/[.:;]$/.test(line) && /^[A-Z]/.test(line) && !line.startsWith("Based on") && !line.startsWith("According to")) {
        htmlParts.push(`<div class="direct-answer-section-header">${formatInlineMarkdown(line)}</div>`);
      } else {
        htmlParts.push(`<p class="direct-answer-para">${formatInlineMarkdown(line)}</p>`);
      }
    }
  }

  if (inList) {
    htmlParts.push("</ul>");
  }

  return htmlParts.join("");
}

/* ==========================================================================
   Typewriter Effect Utilities (Streaming, Typing Loop, Text Animations)
   ========================================================================== */

/**
 * Typewriter Streaming Effect for HTML / Markdown Formatted Text (e.g. AI Chat Answers)
 * Streams words progressively into containerEl with a blinking cursor.
 * Clicking the card immediately completes the stream.
 */
function typewriterStreamHtml(containerEl, fullText, options = {}) {
  if (!containerEl) return;
  const formatFn = options.formatFn || (t => escapeHtml(t));
  const speed = options.speed || 16;
  const scrollContainer = options.scrollContainer || null;
  const onComplete = options.onComplete || null;

  const tokens = fullText.split(/(\s+)/);
  if (tokens.length <= 1) {
    containerEl.innerHTML = formatFn(fullText);
    if (onComplete) onComplete();
    return;
  }

  let tokenIdx = 0;
  let isDone = false;
  const tokensPerTick = tokens.length > 200 ? 3 : (tokens.length > 80 ? 2 : 1);
  const cardParent = containerEl.closest(".direct-answer-card") || containerEl;

  function finishInstantly() {
    if (isDone) return;
    isDone = true;
    if (timerId) clearInterval(timerId);
    containerEl.innerHTML = formatFn(fullText);
    const hint = cardParent.querySelector(".typewriter-skip-hint");
    if (hint) hint.remove();
    cardParent.removeEventListener("click", finishInstantly);
    if (scrollContainer) scrollContainer.scrollTop = scrollContainer.scrollHeight;
    if (onComplete) onComplete();
  }

  cardParent.addEventListener("click", finishInstantly);

  const timerId = setInterval(() => {
    if (isDone) return;
    tokenIdx += tokensPerTick;
    if (tokenIdx >= tokens.length) {
      finishInstantly();
      return;
    }

    const accumulated = tokens.slice(0, tokenIdx).join("");
    containerEl.innerHTML = formatFn(accumulated) + '<span class="typewriter-cursor"></span>';
    if (scrollContainer) {
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
    }
  }, speed);
}

/**
 * Typewriter Streaming Effect for Plain Text (e.g. Quiz Explanations & Notes Intro)
 */
function typewriterStreamText(containerEl, fullText, options = {}) {
  if (!containerEl || !fullText) return;
  const speed = options.speed || 12;
  const onComplete = options.onComplete || null;

  let charIdx = 0;
  let isDone = false;
  const step = fullText.length > 150 ? 2 : 1;

  containerEl.innerHTML = '<span class="typewriter-cursor"></span>';

  function finish() {
    if (isDone) return;
    isDone = true;
    if (timerId) clearInterval(timerId);
    containerEl.textContent = fullText;
    if (onComplete) onComplete();
  }

  containerEl.addEventListener("click", finish);

  const timerId = setInterval(() => {
    if (isDone) return;
    charIdx += step;
    if (charIdx >= fullText.length) {
      finish();
      return;
    }
    containerEl.textContent = fullText.slice(0, charIdx);
    const cursor = document.createElement("span");
    cursor.className = "typewriter-cursor";
    containerEl.appendChild(cursor);
  }, speed);
}

/**
 * Animated Looping Typewriter for Notes Synthesis Loading Status
 * Cycles through status phrases with typing and erasing.
 */
let activeNotesTypewriterStop = null;

function startTypewriterLoop(containerEl, phrases, options = {}) {
  if (!containerEl || !phrases || phrases.length === 0) return () => {};

  let phraseIdx = 0;
  let charIdx = 0;
  let isDeleting = false;
  let isStopped = false;
  let timeoutId = null;

  const typeSpeed = options.typeSpeed || 30;
  const deleteSpeed = options.deleteSpeed || 16;
  const holdTime = options.holdTime || 1600;

  function tick() {
    if (isStopped) return;
    const currentPhrase = phrases[phraseIdx];

    if (!isDeleting) {
      charIdx++;
      containerEl.textContent = currentPhrase.slice(0, charIdx);
      if (charIdx >= currentPhrase.length) {
        isDeleting = true;
        timeoutId = setTimeout(tick, holdTime);
        return;
      }
      timeoutId = setTimeout(tick, typeSpeed);
    } else {
      charIdx--;
      containerEl.textContent = currentPhrase.slice(0, charIdx);
      if (charIdx <= 0) {
        isDeleting = false;
        phraseIdx = (phraseIdx + 1) % phrases.length;
        timeoutId = setTimeout(tick, 280);
        return;
      }
      timeoutId = setTimeout(tick, deleteSpeed);
    }
  }

  timeoutId = setTimeout(tick, 80);

  return function stop() {
    isStopped = true;
    if (timeoutId) clearTimeout(timeoutId);
  };
}

async function handleSemanticSearch(event) {
  event.preventDefault();
  const queryInput = document.getElementById("rag-query");
  const submitButton = document.getElementById("rag-submit");
  const chatMessages = document.getElementById("chat-messages");
  const query = queryInput.value.trim();
  const folderId = activeFolderId;

  if (!query || !folderId) return;

  // Append user message bubble
  const userMsgEl = document.createElement("div");
  userMsgEl.className = "chat-message user";
  userMsgEl.innerHTML = `
    <div class="chat-avatar">You</div>
    <div class="chat-bubble">
      <p>${escapeHtml(query)}</p>
    </div>
  `;
  chatMessages.appendChild(userMsgEl);

  if (!folderChatHistory[folderId]) folderChatHistory[folderId] = [];
  folderChatHistory[folderId].push(userMsgEl);

  queryInput.value = "";
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Append Typing Indicator
  const typingEl = document.createElement("div");
  typingEl.className = "typing-indicator";
  typingEl.id = "chat-typing-indicator";
  typingEl.innerHTML = `
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
  `;
  chatMessages.appendChild(typingEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  if (searchController) searchController.abort();
  const controller = new AbortController();
  searchController = controller;
  submitButton.disabled = true;

  // If no conversation currently active in this folder, create one first
  if (!activeConversationId) {
    try {
      const convRes = await apiFetch(`/api/folders/${folderId}/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_id: folderId, title: query.slice(0, 35) }),
      });
      if (convRes.ok) {
        const convData = await convRes.json();
        activeConversationId = convData.id;
      }
    } catch (e) {
      console.warn("Could not create conversation, falling back:", e);
    }
  }

  try {
    const chatUrl = activeConversationId
      ? `/api/conversations/${activeConversationId}/messages`
      : `/api/search/semantic`;

    const chatBody = activeConversationId
      ? JSON.stringify({ query })
      : JSON.stringify({ query, folder_id: folderId, top_k: 5 });

    const response = await apiFetch(chatUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: chatBody,
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Search failed.");
    if (folderId !== activeFolderId) return;

    // Refresh conversation dropdown in background
    if (activeFolderId) {
      apiFetch(`/api/folders/${activeFolderId}/conversations`)
        .then(r => r.json())
        .then(convs => {
          const selectEl = document.getElementById("chat-conversation-select");
          if (selectEl && convs && convs.length > 0) {
            selectEl.style.display = "inline-block";
            selectEl.innerHTML = "";
            convs.forEach(c => {
              const opt = document.createElement("option");
              opt.value = c.id;
              opt.textContent = `${c.title} (${c.message_count} msgs)`;
              selectEl.appendChild(opt);
            });
            selectEl.value = activeConversationId;
          }
        })
        .catch(() => {});
    }

    // Remove typing indicator
    typingEl.remove();

    // Create Assistant Response Message
    const assistantMsgEl = document.createElement("div");
    assistantMsgEl.className = "chat-message assistant";

    let bubbleContent = "";
    let directAnswerElId = null;

    const citationList = data.citations || data.results || [];

    if (!data.direct_answer && citationList.length === 0) {
      bubbleContent += `<p style="color: var(--text-secondary);">No relevant sources or answers were found for this query in the selected folder.</p>`;
    } else {
      // 1. Direct Answer Card
      if (data.direct_answer) {
        directAnswerElId = `direct-answer-${Date.now()}`;
        const isRefined = data.refinement_metadata && data.refinement_metadata.is_refined;
        const badge = isRefined
          ? `<span style="display:inline-flex; align-items:center; gap:4px; font-size:0.75rem; background:rgba(99, 102, 241, 0.15); color:#818cf8; border:1px solid rgba(99, 102, 241, 0.3); padding:2px 8px; border-radius:12px; margin-left:8px; font-weight:500;">✨ Refined</span>`
          : '';
        bubbleContent += `
          <div class="direct-answer-card">
            <div class="direct-answer-title" style="display:flex; align-items:center; justify-content:space-between;">
              <div style="display:flex; align-items:center;">
                <span>💡 Direct Answer</span>
                ${badge}
              </div>
              <span class="typewriter-skip-hint">Click card to skip typing</span>
            </div>
            <div class="direct-answer-body" id="${directAnswerElId}"></div>
          </div>
        `;
      }

      // 2. Primary Diagram Card
      if (data.primary_figure) {
        const fig = data.primary_figure;
        const figUrl = token ? `${fig.figure_url}?token=${encodeURIComponent(token)}` : fig.figure_url;
        const encodedCap = encodeURIComponent(fig.caption || "");
        bubbleContent += `
          <div class="diagram-card">
            <div class="diagram-header">
              <span>🖼️ Extracted Diagram (Page ${fig.page_number || '1'})</span>
              <span style="font-size:0.75rem; color:var(--text-muted); cursor:pointer;" onclick="openFigureModal('${figUrl}', '${encodedCap}')">Click to zoom</span>
            </div>
            <img class="diagram-preview-img" src="${figUrl}" alt="Diagram" onclick="openFigureModal('${figUrl}', '${encodedCap}')">
            ${fig.caption ? `<div class="diagram-caption">${escapeHtml(fig.caption)}</div>` : ''}
          </div>
        `;
      }

      // 3. Primary Table Card
      if (data.primary_table && data.primary_table.html) {
        const tab = data.primary_table;
        bubbleContent += `
          <div class="table-card">
            <div class="diagram-header" style="margin-bottom:6px;">
              <span>📊 Extracted Table (Page ${tab.page_number || '1'})</span>
            </div>
            ${tab.html}
          </div>
        `;
      }

      // 4. Citation Pills
      if (citationList.length > 0) {
        bubbleContent += `
          <div class="citations-group">
            <span class="citations-label">Sources:</span>
            ${citationList.map(r => `
              <span class="citation-pill" title="${r.section_title ? escapeHtml(r.section_title) : ''}">
                📄 <strong>${escapeHtml(r.source_filename)}</strong>
                ${r.page_number != null ? `· p. ${r.page_number}` : ''}
                ${r.section_title ? `· ${escapeHtml(r.section_title.substring(0, 30))}` : ''}
                <span style="color:var(--status-completed-text); font-weight:600;">(${Math.round(r.relevance_score * 100)}%)</span>
              </span>
            `).join('')}
          </div>
        `;
      }
    }

    assistantMsgEl.innerHTML = `
      <div class="chat-avatar">AI</div>
      <div class="chat-bubble">
        ${bubbleContent}
      </div>
    `;

    chatMessages.appendChild(assistantMsgEl);
    if (folderChatHistory[folderId]) folderChatHistory[folderId].push(assistantMsgEl);

    if (data.direct_answer && directAnswerElId) {
      const answerTarget = document.getElementById(directAnswerElId);
      if (answerTarget) {
        typewriterStreamHtml(answerTarget, data.direct_answer, {
          formatFn: formatDirectAnswerHtml,
          scrollContainer: chatMessages
        });
      }
    }

  } catch (error) {
    if (error.name !== "AbortError") {
      typingEl.remove();
      const errorMsgEl = document.createElement("div");
      errorMsgEl.className = "chat-message assistant";
      errorMsgEl.innerHTML = `
        <div class="chat-avatar">AI</div>
        <div class="chat-bubble">
          <p style="color: var(--status-failed-text);">${escapeHtml(error.message || "Failed to process question. Please try again.")}</p>
        </div>
      `;
      chatMessages.appendChild(errorMsgEl);
      if (folderChatHistory[folderId]) folderChatHistory[folderId].push(errorMsgEl);
    }
  } finally {
    if (searchController === controller) {
      searchController = null;
      submitButton.disabled = false;
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }
}

// Documents Management
async function loadDocuments(folderId) {
  if (!folderId) return;
  try {
    const res = await apiFetch(`/api/folders/${folderId}/documents`);
    if (res.ok) {
      documents = await res.json();
      renderDocumentsTable();
      checkAndStartPolling();
    }
  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}

function renderActiveView() {
  const activeFolder = folders.find((f) => f.id === activeFolderId);
  const title = document.getElementById("active-folder-title");
  const subtitle = document.getElementById("active-folder-subtitle");
  const uploadBtn = document.getElementById("btn-trigger-upload");
  const dropzone = document.getElementById("upload-dropzone");
  const tableContainer = document.getElementById("documents-table-container");
  const emptyState = document.getElementById("empty-state-container");

  if (activeFolder) {
    title.textContent = activeFolder.name;
    subtitle.textContent = `${documents.length} document${documents.length === 1 ? "" : "s"} in this folder`;
    uploadBtn.style.display = "inline-flex";
    dropzone.style.display = "block";
    document.getElementById("rag-panel").style.display = "block";
    document.getElementById("rag-scope-label").textContent = activeFolder.name;
    renderDocumentsTable();
  } else {
    title.textContent = "Select a Folder";
    subtitle.textContent = "Organize and process course documents";
    uploadBtn.style.display = "none";
    dropzone.style.display = "none";
    document.getElementById("rag-panel").style.display = "none";
    tableContainer.style.display = "none";
    emptyState.style.display = "block";
    document.getElementById("empty-state-title").textContent = "No Folder Selected";
    document.getElementById("empty-state-desc").textContent = "Select a folder from the sidebar or create a new one to begin uploading documents.";
  }
}

function renderDocumentsTable() {
  const tableContainer = document.getElementById("documents-table-container");
  const tbody = document.getElementById("document-table-body");
  const emptyState = document.getElementById("empty-state-container");

  if (!activeFolderId) return;

  if (documents.length === 0) {
    tableContainer.style.display = "none";
    emptyState.style.display = "block";
    document.getElementById("empty-state-title").textContent = "No Documents Yet";
    document.getElementById("empty-state-desc").textContent = "Drag & drop a PDF or image into the dropzone above, or click 'Upload Document'.";
    return;
  }

  emptyState.style.display = "none";
  tableContainer.style.display = "block";
  tbody.innerHTML = "";

  documents.forEach((doc) => {
    const tr = document.createElement("tr");

    // File type styling
    const isPdf = doc.file_type.toLowerCase() === "pdf";
    const iconClass = isPdf ? "icon-pdf" : "icon-image";
    const iconLabel = isPdf ? "PDF" : "IMG";

    // Status badge
    let statusClass = "pending";
    let statusLabel = doc.status;
    if (doc.status === "processing") statusClass = "processing";
    else if (doc.status === "completed") statusClass = "completed";
    else if (doc.status === "failed") statusClass = "failed";

    let actionContent = "-";
    if (doc.status === "failed") {
      actionContent = `
        <div style="display: flex; align-items: center; gap: 8px;">
          <button class="btn btn-retry btn-sm" onclick="handleRetry('${doc.id}')" title="Retry document extraction">
            Retry
          </button>
          <span style="font-size: 0.75rem; color: #f87171;" title="${doc.error_message || 'Processing failed'}">
            ⚠️ ${doc.error_message || 'Error'}
          </span>
        </div>
      `;
    }

    tr.innerHTML = `
      <td>
        <div class="doc-name-cell">
          <div class="file-type-icon ${iconClass}">${iconLabel}</div>
          <span title="${doc.original_filename}">${doc.original_filename}</span>
        </div>
      </td>
      <td>${doc.file_type.toUpperCase()}</td>
      <td>${formatBytes(doc.file_size_bytes)}</td>
      <td>${formatDate(doc.created_at)}</td>
      <td>
        <span class="status-badge ${statusClass}">
          <span class="status-dot"></span>
          ${statusLabel}
        </span>
      </td>
      <td>${actionContent}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Background Status Polling
function checkAndStartPolling() {
  const hasActiveJob = documents.some(
    (d) => d.status === "pending" || d.status === "processing"
  );

  if (hasActiveJob && !pollingTimer) {
    pollingTimer = setInterval(async () => {
      if (activeFolderId) {
        try {
          const res = await apiFetch(`/api/folders/${activeFolderId}/documents`);
          if (res.ok) {
            documents = await res.json();
            renderDocumentsTable();
            // Also refresh folder document counts
            const fRes = await apiFetch("/api/folders");
            if (fRes.ok) {
              folders = await fRes.json();
              renderFolders();
            }
            // Stop polling if all jobs are done
            const stillActive = documents.some(
              (d) => d.status === "pending" || d.status === "processing"
            );
            if (!stillActive) {
              clearInterval(pollingTimer);
              pollingTimer = null;
            }
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }
    }, 2500);
  } else if (!hasActiveJob && pollingTimer) {
    clearInterval(pollingTimer);
    pollingTimer = null;
  }
}

// Upload Handling
function triggerFileInput() {
  if (!activeFolderId) {
    showToast("Please select a folder first", "error");
    return;
  }
  document.getElementById("file-input").click();
}

async function handleFileUpload(file) {
  if (!activeFolderId) {
    showToast("Please select a folder first", "error");
    return;
  }
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  showToast(`Uploading "${file.name}"...`);

  try {
    const res = await apiFetch(`/api/folders/${activeFolderId}/documents/upload`, {
      method: "POST",
      body: formData,
    });

    if (res.ok) {
      const data = await res.json();
      showToast(`Uploaded "${file.name}". Ingestion started.`);
      await loadDocuments(activeFolderId);
      await loadFolders();
    } else {
      const err = await res.json();
      showToast(err.detail || "Upload failed", "error");
    }
  } catch (err) {
    showToast("An error occurred during file upload", "error");
  }
}

async function handleRetry(documentId) {
  try {
    showToast("Retrying extraction...");
    const res = await apiFetch(`/api/documents/${documentId}/retry`, {
      method: "POST",
    });
    if (res.ok) {
      showToast("Extraction rescheduled.");
      await loadDocuments(activeFolderId);
    } else {
      const err = await res.json();
      showToast(err.detail || "Retry failed", "error");
    }
  } catch (err) {
    showToast("Error triggering retry", "error");
  }
}

// Setup Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  checkAuth();

  const fileInput = document.getElementById("file-input");
  const dropzone = document.getElementById("upload-dropzone");
  document.getElementById("rag-form").addEventListener("submit", handleSemanticSearch);

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
      fileInput.value = "";
    }
  });

  dropzone.addEventListener("click", () => {
    fileInput.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  // Modal Enter key listener
  document.getElementById("new-folder-name").addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleCreateFolder();
  });
});

/* ==========================================================================
   ATML Practice Quiz Controller Functions
   ========================================================================== */

let currentQuizQuestions = [];
let quizAnswersState = {};
let selectedQuizDocIds = new Set();
let activeQuizTypewriterStop = null;

function openPracticeQuizModal() {
  if (!activeFolderId) {
    showToast("Please select a folder first.", "error");
    return;
  }
  const modal = document.getElementById("practice-quiz-modal");
  if (modal) {
    modal.classList.add("active");
    initQuizUnitSelector();
    loadPracticeQuiz();
  }
}

function closePracticeQuizModal() {
  if (activeQuizTypewriterStop) {
    activeQuizTypewriterStop();
    activeQuizTypewriterStop = null;
  }
  const modal = document.getElementById("practice-quiz-modal");
  if (modal) modal.classList.remove("active");
}

function initQuizUnitSelector() {
  const completedDocs = (documents || []).filter(d => d.status === "completed");
  const selectorBar = document.getElementById("quiz-unit-selector-bar");
  if (!selectorBar) return;

  if (completedDocs.length === 0) {
    selectorBar.style.display = "none";
    return;
  }

  // Ensure selectedQuizDocIds has valid document IDs for active folder
  const validIds = new Set(completedDocs.map(d => d.id));
  const hasValid = Array.from(selectedQuizDocIds).some(id => validIds.has(id));
  if (!hasValid) {
    selectedQuizDocIds = new Set(validIds);
  } else {
    selectedQuizDocIds = new Set(Array.from(selectedQuizDocIds).filter(id => validIds.has(id)));
  }

  renderQuizUnitSelector();
}

function renderQuizUnitSelector() {
  const completedDocs = (documents || []).filter(d => d.status === "completed");
  const selectorBar = document.getElementById("quiz-unit-selector-bar");
  if (!selectorBar || completedDocs.length === 0) {
    if (selectorBar) selectorBar.style.display = "none";
    return;
  }

  selectorBar.style.display = "flex";
  const total = completedDocs.length;
  const count = selectedQuizDocIds.size;
  const allSelected = count === total;

  selectorBar.innerHTML = `
    <div class="notes-unit-header">
      <div class="notes-unit-label">
        <span class="notes-unit-icon">📑</span>
        <span class="notes-unit-heading">Select Units / Files for Quiz:</span>
        <span class="notes-unit-count" id="quiz-unit-count">${count} of ${total} selected</span>
      </div>
      <div class="notes-unit-actions">
        <button type="button" class="notes-unit-toggle-btn" onclick="toggleSelectAllQuizUnits(${!allSelected})">
          ${allSelected ? 'Select First Unit Only' : 'Select All Units'}
        </button>
        <button type="button" class="notes-unit-apply-btn" id="btn-apply-quiz-selection" onclick="applyQuizUnitSelection()" title="Generate practice quiz covering only the selected units">
          ⚡ Generate Quiz for Selected (${count})
        </button>
      </div>
    </div>
    <div class="notes-unit-chips-container">
      ${completedDocs.map(d => {
        const isChecked = selectedQuizDocIds.has(d.id);
        return `
          <label class="notes-unit-chip ${isChecked ? 'active' : ''}" id="chip-quiz-unit-${d.id}" title="${escapeHtml(d.original_filename)}">
            <input type="checkbox" value="${d.id}" ${isChecked ? 'checked' : ''} onchange="onQuizUnitCheckboxChanged('${d.id}', this.checked)">
            <span class="unit-chip-name">📄 ${escapeHtml(d.original_filename)}</span>
          </label>
        `;
      }).join("")}
    </div>
  `;
}

function onQuizUnitCheckboxChanged(docId, isChecked) {
  if (!isChecked) {
    if (selectedQuizDocIds.size <= 1) {
      showToast("⚠️ Selection limit: You must select at least 1 unit file for the quiz.", "warning");
      const input = document.querySelector(`#chip-quiz-unit-${docId} input[type="checkbox"]`);
      if (input) input.checked = true;
      return;
    }
    selectedQuizDocIds.delete(docId);
  } else {
    selectedQuizDocIds.add(docId);
  }
  renderQuizUnitSelector();
}

function toggleSelectAllQuizUnits(selectAll) {
  const completedDocs = (documents || []).filter(d => d.status === "completed");
  if (selectAll) {
    selectedQuizDocIds = new Set(completedDocs.map(d => d.id));
  } else {
    if (completedDocs.length > 0) {
      selectedQuizDocIds = new Set([completedDocs[0].id]);
      showToast("Selected 1 unit (minimum selection limit is 1).", "info");
    }
  }
  renderQuizUnitSelector();
}

async function applyQuizUnitSelection() {
  if (selectedQuizDocIds.size === 0) {
    showToast("Please select at least 1 unit file.", "error");
    return;
  }
  showToast(`🔄 Synthesizing 10 quiz questions for ${selectedQuizDocIds.size} selected unit(s)...`, "info");
  await loadPracticeQuiz(true);
}

async function loadPracticeQuiz(forceRefresh = false) {
  if (!activeFolderId) return;

  const bodyEl = document.getElementById("quiz-modal-body");
  const scoreBadge = document.getElementById("quiz-score-badge");
  const titleEl = document.getElementById("quiz-modal-title");
  const subtitleEl = document.getElementById("quiz-modal-subtitle");

  const currentFolder = folders.find(f => f.id === activeFolderId);
  if (titleEl && currentFolder) {
    titleEl.textContent = `Practice Quiz: ${currentFolder.name}`;
  }
  if (subtitleEl) {
    const unitCount = selectedQuizDocIds ? selectedQuizDocIds.size : 0;
    subtitleEl.textContent = `10 Practice Questions (${unitCount} unit(s) selected)`;
  }

  bodyEl.innerHTML = `
    <div class="quiz-loading-spinner" style="padding: 60px 20px; text-align: center;">
      <div style="font-size: 2.5rem; animation: spin 1s infinite linear; margin-bottom: 16px;">⚙️</div>
      <p style="font-size: 1.05rem; font-weight: 600; color: var(--text-primary); margin-bottom: 6px;">
        Synthesizing 10 Practice Questions...
      </p>
      <div class="typewriter-status-box">
        <span class="typewriter-status-icon">📝</span>
        <span id="quiz-typewriter-status" class="typewriter-status-text"></span>
        <span class="typewriter-cursor"></span>
      </div>
    </div>
  `;
  if (scoreBadge) scoreBadge.textContent = "";
  quizAnswersState = {};

  if (activeQuizTypewriterStop) {
    activeQuizTypewriterStop();
    activeQuizTypewriterStop = null;
  }
  const statusEl = document.getElementById("quiz-typewriter-status");
  if (statusEl) {
    activeQuizTypewriterStop = startTypewriterLoop(statusEl, [
      "Mining fact anchors from selected unit materials...",
      "Formulating conceptual question stems (FLAN-T5)...",
      "Optimizing vector manifold distractors (all-MiniLM)...",
      "Verifying page citations and pedagogical explanations...",
      "Assembling 10 practice questions..."
    ]);
  }

  try {
    let url = `/api/folders/${activeFolderId}/quiz?num_questions=10`;
    if (selectedQuizDocIds && selectedQuizDocIds.size > 0) {
      url += `&doc_ids=${encodeURIComponent(Array.from(selectedQuizDocIds).join(","))}`;
    }

    const res = await apiFetch(url);
    if (activeQuizTypewriterStop) {
      activeQuizTypewriterStop();
      activeQuizTypewriterStop = null;
    }

    if (!res.ok) {
      const err = await res.json();
      bodyEl.innerHTML = `
        <div style="text-align: center; padding: 30px; color: var(--text-secondary);">
          <p style="color: #ef4444; font-weight: 600;">⚠️ ${escapeHtml(err.detail || "Failed to generate quiz.")}</p>
          <p style="font-size: 0.85rem; margin-top: 8px;">Ensure you have selected completed documents in this folder.</p>
        </div>
      `;
      return;
    }

    const data = await res.json();
    currentQuizQuestions = data.questions || [];

    if (currentQuizQuestions.length === 0) {
      bodyEl.innerHTML = `
        <div style="text-align: center; padding: 40px 20px; color: var(--text-secondary);">
          <span style="font-size: 2.5rem; display: block; margin-bottom: 12px;">📚</span>
          <p style="font-weight: 600; font-size: 1rem; color: var(--text-primary);">No Course Material Available Yet</p>
          <p style="font-size: 0.85rem; margin-top: 6px;">Upload course notes or slides (PDF) to this folder to generate practice questions.</p>
        </div>
      `;
      return;
    }

    renderQuizQuestions();
  } catch (err) {
    if (activeQuizTypewriterStop) {
      activeQuizTypewriterStop();
      activeQuizTypewriterStop = null;
    }
    bodyEl.innerHTML = `
      <div style="text-align: center; padding: 30px; color: #ef4444;">
        <p>⚠️ Error communicating with quiz engine: ${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function renderQuizQuestions() {
  const bodyEl = document.getElementById("quiz-modal-body");
  const scoreBadge = document.getElementById("quiz-score-badge");
  let html = "";

  currentQuizQuestions.forEach((q, qIndex) => {
    const pageCitation = q.page_number ? `Page ${q.page_number}` : 'Notes';
    html += `
      <div class="quiz-question-card" id="card-${q.id}">
        <div class="quiz-q-meta">
          <span class="quiz-q-badge">Question ${qIndex + 1} of ${currentQuizQuestions.length}</span>
          <span class="quiz-q-source">📖 ${escapeHtml(q.source_document || 'Course Material')} (${pageCitation})</span>
        </div>
        <div class="quiz-q-title">${escapeHtml(q.question)}</div>
        <div class="quiz-options-group" id="options-${q.id}">
    `;

    q.options.forEach((optText, optIdx) => {
      html += `
        <button
          type="button"
          class="quiz-option-btn"
          id="btn-${q.id}-${optIdx}"
          onclick="handleSelectQuizOption('${q.id}', ${optIdx}, ${q.correct_option_index})"
        >
          <span>${escapeHtml(optText)}</span>
        </button>
      `;
    });

    html += `
        </div>
        <div class="quiz-explanation-box" id="explanation-${q.id}" style="display: none;"></div>
      </div>
    `;
  });

  bodyEl.innerHTML = html;
  updateQuizScoreBadge();
}

function handleSelectQuizOption(questionId, selectedIdx, correctIdx) {
  if (quizAnswersState[questionId] !== undefined) return; // Already answered

  const isCorrect = selectedIdx === correctIdx;
  quizAnswersState[questionId] = isCorrect;

  const q = currentQuizQuestions.find(item => item.id === questionId);
  const selectedBtn = document.getElementById(`btn-${questionId}-${selectedIdx}`);
  const correctBtn = document.getElementById(`btn-${questionId}-${correctIdx}`);
  const expBox = document.getElementById(`explanation-${questionId}`);

  // Highlight selected button
  if (selectedBtn) {
    selectedBtn.classList.add(isCorrect ? "selected-correct" : "selected-incorrect");
  }

  // Highlight correct button if incorrect
  if (!isCorrect && correctBtn) {
    correctBtn.classList.add("selected-correct");
  }

  // Disable all options for this question
  if (q && q.options) {
    q.options.forEach((_, idx) => {
      const b = document.getElementById(`btn-${questionId}-${idx}`);
      if (b) b.disabled = true;
    });
  }

  // Show explanation with typewriter streaming
  if (expBox && q) {
    expBox.style.display = "block";
    expBox.className = `quiz-explanation-box ${isCorrect ? 'correct' : 'incorrect'}`;
    const expTextId = `exp-text-${questionId}`;
    expBox.innerHTML = `
      <div style="font-weight: 600; margin-bottom: 4px;">
        ${isCorrect ? '✅ Correct Answer!' : '❌ Incorrect'}
      </div>
      <div id="${expTextId}"></div>
    `;
    const expTextEl = document.getElementById(expTextId);
    if (expTextEl) {
      typewriterStreamText(expTextEl, q.explanation || '', { speed: 8 });
    }
  }

  updateQuizScoreBadge();
}

function updateQuizScoreBadge() {
  const scoreBadge = document.getElementById("quiz-score-badge");
  if (!scoreBadge) return;

  const totalAnswered = Object.keys(quizAnswersState).length;
  const correctCount = Object.values(quizAnswersState).filter(Boolean).length;
  const totalQuestions = currentQuizQuestions.length;

  if (totalAnswered > 0) {
    scoreBadge.textContent = `Score: ${correctCount} / ${totalQuestions} (${Math.round((correctCount / totalQuestions) * 100)}%)`;
  } else {
    scoreBadge.textContent = `${totalQuestions} questions synthesized`;
  }
}

// ==========================================
// Comprehensive Study Notes Logic
// ==========================================
// Comprehensive Study Notes Logic
// ==========================================
let currentNotesData = null;
let selectedUnitDocIds = new Set();

function openNotesModal() {
  if (!activeFolderId) {
    showToast("Please select a folder first.", "error");
    return;
  }
  const modal = document.getElementById("notes-reader-modal");
  if (modal) {
    modal.classList.add("active");
    initNotesUnitSelector();
    loadNotes();
  }
}

function closeNotesModal() {
  if (activeNotesTypewriterStop) {
    activeNotesTypewriterStop();
    activeNotesTypewriterStop = null;
  }
  const modal = document.getElementById("notes-reader-modal");
  if (modal) modal.classList.remove("active");
}

function initNotesUnitSelector() {
  const completedDocs = (documents || []).filter(d => d.status === "completed");
  const selectorBar = document.getElementById("notes-unit-selector-bar");
  if (!selectorBar) return;

  if (completedDocs.length === 0) {
    selectorBar.style.display = "none";
    return;
  }

  // Ensure selectedUnitDocIds has valid document IDs for active folder
  const validIds = new Set(completedDocs.map(d => d.id));
  const hasValid = Array.from(selectedUnitDocIds).some(id => validIds.has(id));
  if (!hasValid) {
    selectedUnitDocIds = new Set(validIds);
  } else {
    selectedUnitDocIds = new Set(Array.from(selectedUnitDocIds).filter(id => validIds.has(id)));
  }

  renderNotesUnitSelector();
}

function renderNotesUnitSelector() {
  const completedDocs = (documents || []).filter(d => d.status === "completed");
  const selectorBar = document.getElementById("notes-unit-selector-bar");
  if (!selectorBar || completedDocs.length === 0) {
    if (selectorBar) selectorBar.style.display = "none";
    return;
  }

  selectorBar.style.display = "flex";
  const total = completedDocs.length;
  const count = selectedUnitDocIds.size;
  const allSelected = count === total;

  selectorBar.innerHTML = `
    <div class="notes-unit-header">
      <div class="notes-unit-label">
        <span class="notes-unit-icon">📑</span>
        <span class="notes-unit-heading">Select Units / Files:</span>
        <span class="notes-unit-count" id="notes-unit-count">${count} of ${total} selected</span>
      </div>
      <div class="notes-unit-actions">
        <button type="button" class="notes-unit-toggle-btn" onclick="toggleSelectAllUnits(${!allSelected})">
          ${allSelected ? 'Select First Unit Only' : 'Select All Units'}
        </button>
        <button type="button" class="notes-unit-apply-btn" id="btn-apply-unit-selection" onclick="applyUnitSelection()" title="Re-synthesize study notes covering only the selected units">
          ⚡ Generate for Selected (${count})
        </button>
      </div>
    </div>
    <div class="notes-unit-chips-container">
      ${completedDocs.map(d => {
        const isChecked = selectedUnitDocIds.has(d.id);
        return `
          <label class="notes-unit-chip ${isChecked ? 'active' : ''}" id="chip-unit-${d.id}" title="${escapeHtml(d.original_filename)}">
            <input type="checkbox" value="${d.id}" ${isChecked ? 'checked' : ''} onchange="onUnitCheckboxChanged('${d.id}', this.checked)">
            <span class="unit-chip-name">📄 ${escapeHtml(d.original_filename)}</span>
          </label>
        `;
      }).join("")}
    </div>
  `;
}

function onUnitCheckboxChanged(docId, isChecked) {
  if (!isChecked) {
    if (selectedUnitDocIds.size <= 1) {
      showToast("⚠️ Selection limit: You must select at least 1 unit file.", "warning");
      const input = document.querySelector(`#chip-unit-${docId} input[type="checkbox"]`);
      if (input) input.checked = true;
      return;
    }
    selectedUnitDocIds.delete(docId);
  } else {
    selectedUnitDocIds.add(docId);
  }
  renderNotesUnitSelector();
}

function toggleSelectAllUnits(selectAll) {
  const completedDocs = (documents || []).filter(d => d.status === "completed");
  if (selectAll) {
    selectedUnitDocIds = new Set(completedDocs.map(d => d.id));
  } else {
    if (completedDocs.length > 0) {
      selectedUnitDocIds = new Set([completedDocs[0].id]);
      showToast("Selected 1 unit (minimum selection limit is 1).", "info");
    }
  }
  renderNotesUnitSelector();
}

async function applyUnitSelection() {
  if (selectedUnitDocIds.size === 0) {
    showToast("Please select at least 1 unit file.", "error");
    return;
  }
  showToast(`🔄 Synthesizing study notes for ${selectedUnitDocIds.size} selected unit(s)...`, "info");
  await loadNotes(true);
}

async function refreshNotes() {
  if (!activeFolderId) return;
  showToast("🔄 Re-synthesizing study notes from lecture materials...", "success");
  await loadNotes(true);
}

async function loadNotes(forceRefresh = false) {
  if (!activeFolderId) return;

  const bodyEl = document.getElementById("notes-modal-body");
  const titleEl = document.getElementById("notes-modal-title");
  const subtitleEl = document.getElementById("notes-modal-subtitle");

  const currentFolder = folders.find(f => f.id === activeFolderId);
  if (titleEl && currentFolder) {
    titleEl.textContent = `Study Guide: ${currentFolder.name}`;
  }
  if (subtitleEl) {
    subtitleEl.textContent = forceRefresh
      ? "Re-synthesizing comprehensive course notes from selected lecture units..."
      : "Loading comprehensive course notes (Instant cache or synthesizing)...";
  }

  bodyEl.innerHTML = `
    <div class="quiz-loading-spinner" style="padding: 60px 20px; text-align: center;">
      <div style="font-size: 2.5rem; animation: spin 1s infinite linear; margin-bottom: 16px;">📚</div>
      <p style="font-size: 1.05rem; font-weight: 600; color: var(--text-primary); margin-bottom: 6px;">
        ${forceRefresh ? "Synthesizing Selected Units..." : "Loading Study Guide..."}
      </p>
      <div class="typewriter-status-box">
        <span class="typewriter-status-icon">⚡</span>
        <span id="notes-typewriter-status" class="typewriter-status-text"></span>
        <span class="typewriter-cursor"></span>
      </div>
    </div>
  `;

  if (activeNotesTypewriterStop) {
    activeNotesTypewriterStop();
    activeNotesTypewriterStop = null;
  }
  const statusEl = document.getElementById("notes-typewriter-status");
  if (statusEl) {
    activeNotesTypewriterStop = startTypewriterLoop(statusEl, [
      "Analyzing lecture materials & selected units...",
      "Mining core conceptual anchors & definitions...",
      "Generating cross-unit comparison matrices...",
      "Structuring high-yield university exam takeaways...",
      "Polishing comprehensive study guide..."
    ]);
  }

  try {
    let url = `/api/folders/${activeFolderId}/notes`;
    const queryParts = [];
    if (forceRefresh) queryParts.push("refresh=true");
    if (selectedUnitDocIds && selectedUnitDocIds.size > 0) {
      queryParts.push(`doc_ids=${encodeURIComponent(Array.from(selectedUnitDocIds).join(","))}`);
    }
    if (queryParts.length > 0) {
      url += `?${queryParts.join("&")}`;
    }

    const res = await apiFetch(url);
    if (!res.ok) {
      const err = await res.json();
      if (activeNotesTypewriterStop) {
        activeNotesTypewriterStop();
        activeNotesTypewriterStop = null;
      }
      bodyEl.innerHTML = `
        <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
          <p style="color: #ef4444; font-weight: 600; font-size: 1rem;">⚠️ ${escapeHtml(err.detail || "Failed to generate study guide.")}</p>
          <p style="font-size: 0.85rem; margin-top: 8px;">Ensure you have selected completed units in this folder.</p>
        </div>
      `;
      return;
    }

    const data = await res.json();
    currentNotesData = data;

    if (subtitleEl) {
      subtitleEl.textContent = `Generated on ${data.generated_at} · ${data.documents_analyzed.length} document(s) included`;
    }

    renderNotes(data);
  } catch (error) {
    if (activeNotesTypewriterStop) {
      activeNotesTypewriterStop();
      activeNotesTypewriterStop = null;
    }
    console.error("Notes generation error:", error);
    bodyEl.innerHTML = `
      <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
        <p style="color: #ef4444; font-weight: 600;">⚠️ Error connecting to server.</p>
        <p style="font-size: 0.85rem; margin-top: 8px;">${escapeHtml(error.message || "Please try again later.")}</p>
      </div>
    `;
  }
}

function renderNotes(data) {
  if (activeNotesTypewriterStop) {
    activeNotesTypewriterStop();
    activeNotesTypewriterStop = null;
  }
  const bodyEl = document.getElementById("notes-modal-body");
  if (!bodyEl || !data) return;

  const docsBadge = data.documents_analyzed.map(d => `<span class="notes-doc-badge">📄 ${escapeHtml(d)}</span>`).join("");

  let sectionsHtml = "";
  if (data.sections && data.sections.length > 0) {
    sectionsHtml = data.sections.map((sec, idx) => {
      const sectionHtml = renderMarkdownToHtml(sec.content_markdown);
      return `
        <details class="notes-section-card notes-stream-enter" open id="${escapeHtml(sec.id)}" style="animation-delay: ${idx * 50}ms;">
          <summary class="notes-section-header">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span class="notes-sec-icon">${sec.icon || '📌'}</span>
              <span class="notes-sec-title">${escapeHtml(sec.title)}</span>
            </div>
            <span class="notes-sec-toggle">▾</span>
          </summary>
          <div class="notes-section-content">
            ${sectionHtml}
          </div>
        </details>
      `;
    }).join("");
  } else {
    sectionsHtml = `<div class="notes-section-content">${renderMarkdownToHtml(data.markdown_content)}</div>`;
  }

  bodyEl.innerHTML = `
    <div class="notes-container">
      <div class="notes-meta-banner">
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 6px; font-weight: 500;">Analyzed Course Materials:</div>
        <div style="display: flex; flex-wrap: wrap; gap: 6px;">
          ${docsBadge}
        </div>
        <div id="notes-intro-summary" style="margin-top: 10px; font-size: 0.84rem; color: #cbd5e1; line-height: 1.5; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
        </div>
      </div>
      <div class="notes-sections-list">
        ${sectionsHtml}
      </div>
    </div>
  `;

  const introEl = document.getElementById("notes-intro-summary");
  if (introEl) {
    const totalDocs = data.documents_analyzed ? data.documents_analyzed.length : 1;
    const totalSecs = data.sections ? data.sections.length : 0;
    const introText = `✨ Comprehensive study notes synthesized across ${totalDocs} unit(s) with ${totalSecs} topic modules. Review all sections below for exam-oriented conceptual clarity.`;
    typewriterStreamText(introEl, introText, { speed: 12 });
  }
}

function copyNotesToClipboard() {
  if (!currentNotesData || !currentNotesData.markdown_content) {
    showToast("No notes content to copy.", "error");
    return;
  }
  navigator.clipboard.writeText(currentNotesData.markdown_content).then(() => {
    showToast("📋 Study notes copied to clipboard in Markdown format!", "success");
  }).catch(err => {
    console.error("Clipboard copy failed:", err);
    showToast("Failed to copy notes to clipboard.", "error");
  });
}


function downloadNotesMarkdown() {
  if (!currentNotesData || !currentNotesData.markdown_content) {
    showToast("No notes content available to download.", "error");
    return;
  }
  const blob = new Blob([currentNotesData.markdown_content], { type: "text/markdown;charset=utf-8" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.style.display = "none";
  a.href = url;
  const safeName = (currentNotesData.folder_name || "Course").replace(/[^a-zA-Z0-9_-]/g, "_");
  a.download = `Study_Guide_${safeName}.md`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  a.remove();
  showToast("📝 Markdown file downloaded!", "success");
}

function printNotesPDF() {
  if (!currentNotesData) {
    showToast("Please wait for notes to finish generating.", "error");
    return;
  }
  // Expand all details cards so print preview captures all pages and sections without clipping
  const sectionCards = document.querySelectorAll("#notes-modal-body details");
  sectionCards.forEach(el => el.setAttribute("open", "true"));
  window.print();
}

function renderMarkdownToHtml(md) {
  if (!md) return "";
  let html = md;

  // Escape basic HTML except intentional formatting
  html = html
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Fenced Code Blocks (```lang ... ```)
  html = html.replace(/```(?:[a-zA-Z0-9_\-]+)?\r?\n([\s\S]*?)```/g, function(match, code) {
    return '<pre class="notes-code-block"><code>' + code.trim() + '</code></pre>';
  });

  // Headers
  html = html.replace(/^#### (.*?)$/gm, '<h4 class="notes-h4">$1</h4>');
  html = html.replace(/^### (.*?)$/gm, '<h3 class="notes-h3">$1</h3>');
  html = html.replace(/^## (.*?)$/gm, '<h2 class="notes-h2">$1</h2>');
  html = html.replace(/^# (.*?)$/gm, '<h1 class="notes-h1">$1</h1>');

  // Blockquotes (Callouts)
  html = html.replace(/^&gt; (.*?)$/gm, '<blockquote class="notes-blockquote">$1</blockquote>');

  // Bold & Italic & Inline Code
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  html = html.replace(/`([^`]+)`/g, '<code class="notes-inline-code">$1</code>');

  // Horizontal Rules
  html = html.replace(/^---$/gm, '<hr class="notes-hr">');

  // Tables
  html = html.replace(/((?:\|[^\n]+\|\r?\n)+)/g, function(tableMatch) {
    const lines = tableMatch.trim().split("\n");
    if (lines.length < 2) return tableMatch;

    let tableHtml = '<div class="notes-table-container"><table class="notes-table">';
    const headerCells = lines[0].split("|").slice(1, -1);
    tableHtml += '<thead><tr>' + headerCells.map(c => `<th>${c.trim()}</th>`).join('') + '</tr></thead><tbody>';

    for (let i = 2; i < lines.length; i++) {
      const rowCells = lines[i].split("|").slice(1, -1);
      if (rowCells.length > 0) {
        tableHtml += '<tr>' + rowCells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
      }
    }
    tableHtml += '</tbody></table></div>';
    return tableHtml;
  });

  // Numbered Lists (with optional indentation)
  html = html.replace(/^(\s*)(\d+)\.\s+(.*?)$/gm, function(m, indent, num, text) {
    const ml = indent.length > 0 ? ' style="margin-left: ' + (indent.length * 10) + 'px;"' : '';
    return '<div class="notes-list-item numbered"' + ml + '><span class="list-num">' + num + '.</span> <span>' + text + '</span></div>';
  });

  // Bullet Lists (with optional indentation and support for - and *)
  html = html.replace(/^(\s*)[-*]\s+(.*?)$/gm, function(m, indent, text) {
    const ml = indent.length > 0 ? ' style="margin-left: ' + (indent.length * 10) + 'px;"' : '';
    return '<div class="notes-list-item bullet"' + ml + '><span class="list-bullet">•</span> <span>' + text + '</span></div>';
  });

  // Paragraphs
  const blocks = html.split(/\n\s*\n/);
  html = blocks.map(b => {
    b = b.trim();
    if (!b) return "";
    if (b.startsWith("<h") || b.startsWith("<div") || b.startsWith("<blockquote") || b.startsWith("<table") || b.startsWith("<hr") || b.startsWith("<pre")) {
      return b;
    }
    return `<p class="notes-p">${b.replace(/\n/g, "<br>")}</p>`;
  }).join("\n");

  return html;
}

