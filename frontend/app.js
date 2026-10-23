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
let activeDocumentId = null;
let activeView = "home"; // 'home', 'chat', 'notes', 'quiz'
let isSidebarCollapsed = false;
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
    // No demo user: prompt login screen first!
    updateAuthUI();
    openAuthModal();
    return;
  }
  try {
    const res = await apiFetch("/api/auth/me");
    if (res.ok) {
      currentUser = await res.json();
      updateAuthUI();
      await loadFolders();
    } else {
      localStorage.removeItem("nalanda_token");
      token = null;
      currentUser = null;
      updateAuthUI();
      openAuthModal();
    }
  } catch (err) {
    console.error("Auth check failed:", err);
    localStorage.removeItem("nalanda_token");
    token = null;
    currentUser = null;
    updateAuthUI();
    openAuthModal();
  }
}

function updateAuthUI() {
  const badge = document.getElementById("user-profile-badge");
  const avatar = document.getElementById("user-avatar-text");
  const name = document.getElementById("user-display-name");
  const btnLogout = document.getElementById("btn-logout");
  const greetingEl = document.getElementById("home-greeting-heading");

  if (currentUser) {
    if (badge) badge.style.display = "flex";
    if (avatar) avatar.textContent = (currentUser.username || "U")[0].toUpperCase();
    if (name) name.textContent = currentUser.username;
    if (btnLogout) btnLogout.style.display = "inline-flex";
    if (greetingEl) greetingEl.textContent = `What are we studying today, ${currentUser.username}?`;
  } else {
    if (badge) badge.style.display = "none";
    if (btnLogout) btnLogout.style.display = "none";
    if (greetingEl) greetingEl.textContent = "What are we studying today?";
  }
}

function openAuthModal() {
  const modal = document.getElementById("auth-modal");
  if (modal) {
    modal.style.display = "flex";
    modal.classList.add("active");
    const closeBtn = document.getElementById("auth-modal-close-btn");
    if (closeBtn) {
      closeBtn.style.display = currentUser ? "block" : "none";
    }
    setTimeout(() => {
      const emailInput = document.getElementById("auth-email");
      if (emailInput) emailInput.focus();
    }, 100);
  }
}

function closeAuthModal() {
  if (!currentUser) {
    showToast("Please sign in or create an account to access Nalanda.", "error");
    return;
  }
  const modal = document.getElementById("auth-modal");
  if (modal) {
    modal.style.display = "none";
    modal.classList.remove("active");
  }
}

function toggleAuthMode(event) {
  if (event) event.preventDefault();
  isRegisterMode = !isRegisterMode;
  const title = document.getElementById("auth-modal-title");
  const btn = document.getElementById("auth-submit-btn") || document.getElementById("btn-auth-submit");
  const toggleLink = document.getElementById("auth-toggle-btn") || document.getElementById("auth-toggle-link");
  const usernameGroup = document.getElementById("username-group");

  if (isRegisterMode) {
    if (title) title.textContent = "Create an Account";
    if (btn) btn.textContent = "Register";
    if (toggleLink) toggleLink.textContent = "Already have an account? Sign in";
    if (usernameGroup) usernameGroup.style.display = "block";
  } else {
    if (title) title.textContent = "Sign In to Nalanda AI";
    if (btn) btn.textContent = "Sign In";
    if (toggleLink) toggleLink.textContent = "Need an account? Register here";
    if (usernameGroup) usernameGroup.style.display = "none";
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
  localStorage.removeItem("nalanda_active_folder_id");
  localStorage.removeItem("nalanda_active_conversation_id");
  localStorage.removeItem("nalanda_active_modal");
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
  openAuthModal();
}

// Folders Management
async function loadFolders() {
  try {
    const res = await apiFetch("/api/folders");
    if (res.ok) {
      folders = await res.json();
      renderFolders();

      const savedFolderId = localStorage.getItem("nalanda_active_folder_id");
      const folderExists = savedFolderId && folders.some(f => f.id === savedFolderId);

      // If saved folder exists, restore it. Otherwise fallback to activeFolderId or first folder.
      if (folderExists) {
        await selectFolder(savedFolderId);
      } else if (!activeFolderId && folders.length > 0) {
        await selectFolder(folders[0].id);
      } else if (activeFolderId && folders.some(f => f.id === activeFolderId)) {
        await selectFolder(activeFolderId);
      } else if (folders.length > 0) {
        await selectFolder(folders[0].id);
      } else {
        renderActiveView();
      }

      // Restore last action modal (e.g. Study Notes or Practice Quiz) if it was open before reload
      restoreActiveModalState();
    }
  } catch (err) {
    console.error("Failed to load folders:", err);
  }
}

function restoreActiveModalState() {
  const currentRoute = getCurrentRoute();
  switchView(currentRoute, false);
}

function renderFolders() {
  const container = document.getElementById("folder-list-container");
  if (!container) return;
  container.innerHTML = "";

  if (folders.length === 0) {
    container.innerHTML = `
      <div class="p-3 text-center text-outline text-label-sm">
        No folders yet.
        <button type="button" class="text-primary hover:underline font-medium block mx-auto mt-1 cursor-pointer" onclick="openCreateFolderModal()">Create a folder</button>
      </div>
    `;
    return;
  }

  folders.forEach((folder) => {
    const isFolderActive = folder.id === activeFolderId;
    const item = document.createElement("div");
    item.className = `folder-tree-item ${isFolderActive ? "active" : ""}`;
    item.id = `folder-tree-${folder.id}`;

    const arrowIcon = isFolderActive ? "arrow_drop_down" : "arrow_right";
    const folderIcon = isFolderActive ? "folder_open" : "folder";
    const folderIconColor = isFolderActive ? "text-primary" : "text-outline";

    let docsTreeHtml = "";
    if (isFolderActive && documents && documents.length > 0) {
      docsTreeHtml = `
        <div class="flex flex-col pl-6 pr-1 gap-0.5 border-l border-outline-variant/60 ml-4 my-1">
          ${documents.map(doc => {
            const isDocActive = doc.id === activeDocumentId;
            const isPdf = doc.file_type.toLowerCase() === "pdf";
            const docIcon = isPdf ? "picture_as_pdf" : "description";
            const iconColor = isPdf ? "text-error" : "text-primary";
            const activeBg = isDocActive ? "bg-surface-container-high text-primary font-medium" : "text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high";
            return `
              <div class="folder-doc-link flex items-center justify-between py-1 px-2 rounded ${activeBg} text-body-sm truncate transition-colors cursor-pointer" onclick="selectDocument('${doc.id}', event)">
                <div class="flex items-center gap-2 truncate flex-1 min-w-0 mr-1">
                  <span class="material-symbols-outlined text-[16px] ${iconColor} shrink-0">${docIcon}</span>
                  <span class="truncate" title="${escapeHtml(doc.original_filename)}">${escapeHtml(doc.original_filename)}</span>
                </div>
                <div class="flex items-center gap-1 shrink-0">
                  ${isDocActive ? '<span class="w-1.5 h-1.5 rounded-full bg-tertiary-fixed-dim" title="Currently active in context scope"></span>' : ''}
                  <button type="button" class="doc-delete-btn text-outline hover:text-error" title="Delete file" onclick="event.stopPropagation(); handleDeleteDocument('${doc.id}', '${escapeHtml(doc.original_filename).replace(/'/g, "\\'")}')">
                    <span class="material-symbols-outlined text-[14px]">delete</span>
                  </button>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      `;
    }

    item.innerHTML = `
      <div class="folder-tree-header" onclick="selectFolder('${folder.id}')">
        <div class="flex items-center gap-2 truncate flex-1 min-w-0">
          <span class="material-symbols-outlined text-outline-variant text-[16px]">${arrowIcon}</span>
          <span class="material-symbols-outlined ${folderIconColor} text-[18px]">${folderIcon}</span>
          <span class="text-body-sm font-medium truncate">${escapeHtml(folder.name)}</span>
        </div>
        <div class="flex items-center gap-1.5 ml-2 shrink-0">
          <span class="text-label-sm text-outline px-1.5 py-0.2 rounded-full bg-surface-container">${folder.document_count || 0}</span>
          <button type="button" class="folder-delete-btn" title="Delete folder" onclick="confirmDeleteFolder('${folder.id}', '${escapeHtml(folder.name).replace(/'/g, "\\'")}', event)">
            <span class="material-symbols-outlined text-[15px]">delete</span>
          </button>
        </div>
      </div>
      ${docsTreeHtml}
    `;

    container.appendChild(item);
  });
}

function selectDocument(docId, event) {
  if (event) event.stopPropagation();
  activeDocumentId = docId;
  const doc = documents.find(d => d.id === docId);
  renderActiveView();
  renderFolders();
  if (doc) {
    showToast(`Active document: ${doc.original_filename}`);
  }
}

function openCreateFolderModal() {
  const modal = document.getElementById("create-folder-modal");
  if (!modal) return;
  const input = document.getElementById("new-folder-name");
  if (input) input.value = "";
  modal.style.display = "flex";
  modal.classList.add("active");
  setTimeout(() => {
    if (input) input.focus();
  }, 100);
}

function closeCreateFolderModal() {
  const modal = document.getElementById("create-folder-modal");
  if (!modal) return;
  modal.style.display = "none";
  modal.classList.remove("active");
}

let isCreatingFolder = false;

async function handleCreateFolder() {
  if (isCreatingFolder) return;

  const nameInput = document.getElementById("new-folder-name");
  if (!nameInput) return;
  const name = nameInput.value.trim();
  if (!name) {
    showToast("Please enter a folder name", "error");
    nameInput.focus();
    return;
  }

  isCreatingFolder = true;
  const submitBtn = document.getElementById("btn-create-folder-submit");
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = "Creating...";
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
      activeFolderId = newFolder.id;
      localStorage.setItem("nalanda_active_folder_id", newFolder.id);
      await loadFolders();
      await selectFolder(newFolder.id);
    } else {
      const data = await res.json();
      showToast(data.detail || "Failed to create folder", "error");
    }
  } catch (err) {
    console.error("Error creating folder:", err);
    showToast("Error creating folder", "error");
  } finally {
    isCreatingFolder = false;
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = "Create Folder";
    }
  }
}

let folderToDelete = null;

function confirmDeleteFolder(folderId, folderName, event) {
  if (event) event.stopPropagation();
  folderToDelete = { id: folderId, name: folderName };

  const modal = document.getElementById("delete-folder-modal");
  const nameEl = document.getElementById("delete-folder-modal-name");
  if (nameEl) nameEl.textContent = folderName;
  if (modal) {
    modal.style.display = "flex";
    modal.classList.add("active");
  }
}

function closeDeleteFolderModal() {
  folderToDelete = null;
  const modal = document.getElementById("delete-folder-modal");
  if (modal) {
    modal.style.display = "none";
    modal.classList.remove("active");
  }
}

async function handleDeleteFolder() {
  if (!folderToDelete || !folderToDelete.id) return;
  const { id: folderId, name: folderName } = folderToDelete;

  const btn = document.getElementById("btn-delete-folder-confirm");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Deleting...";
  }

  try {
    const res = await apiFetch(`/api/folders/${folderId}`, {
      method: "DELETE",
    });

    if (res.ok) {
      closeDeleteFolderModal();
      showToast(`Deleted folder "${folderName}"`);

      if (activeFolderId === folderId) {
        activeFolderId = null;
        localStorage.removeItem("nalanda_active_folder_id");
      }

      await loadFolders();

      if (folders.length > 0) {
        if (!activeFolderId || !folders.some(f => f.id === activeFolderId)) {
          await selectFolder(folders[0].id);
        }
      } else {
        renderActiveView();
      }
    } else {
      const data = await res.json();
      showToast(data.detail || "Failed to delete folder", "error");
    }
  } catch (err) {
    console.error("Error deleting folder:", err);
    showToast("Error deleting folder", "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Delete Folder";
    }
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

        // Restore saved conversation for this folder if exists
        const savedConvId = localStorage.getItem(`nalanda_active_conv_${folderId}`) || localStorage.getItem("nalanda_active_conversation_id");

        if (savedConvId === "new") {
          activeConversationId = null;
          selectEl.selectedIndex = -1;
          renderChatWelcome();
          return;
        }

        const matchingConv = folderConversations.find(c => c.id === savedConvId);

        if (matchingConv) {
          activeConversationId = matchingConv.id;
          selectEl.value = activeConversationId;
          localStorage.setItem("nalanda_active_conversation_id", activeConversationId);
          localStorage.setItem(`nalanda_active_conv_${folderId}`, activeConversationId);
          await loadConversationMessages(activeConversationId);
        } else {
          // If no specific matching conversation was active or user chose new chat, stay in clean new chat state
          activeConversationId = null;
          selectEl.selectedIndex = -1;
          renderChatWelcome();
        }
      } else {
        selectEl.style.display = "none";
        activeConversationId = null;
        localStorage.setItem("nalanda_active_conversation_id", "new");
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

function getIlayasAvatarHtml() {
  return `
    <div class="chat-avatar" title="Ilayas AI">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" />
        <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" transform="rotate(72 12 12)" />
        <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" transform="rotate(144 12 12)" />
        <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" transform="rotate(216 12 12)" />
        <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" transform="rotate(288 12 12)" />
        <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" />
        <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" transform="rotate(72 12 12)" />
        <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" transform="rotate(144 12 12)" />
        <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" transform="rotate(216 12 12)" />
        <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" transform="rotate(288 12 12)" />
        <circle cx="12" cy="12" r="2.1" fill="#818cf8" />
        <circle cx="12" cy="12" r="1.1" fill="#e0e7ff" />
      </svg>
    </div>
  `;
}

function renderChatWelcome() {
  const container = document.getElementById("chat-messages");
  if (!container) return;
  container.innerHTML = `
    <div class="ilayas-welcome-card">
      <div class="ilayas-badge-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" />
          <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" transform="rotate(72 12 12)" />
          <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" transform="rotate(144 12 12)" />
          <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" transform="rotate(216 12 12)" />
          <path d="M12 4.2C10.3 6.4 9.7 9.3 11.5 11.7C11.7 12 12.3 12 12.5 11.7C14.3 9.3 13.7 6.4 12 4.2Z" fill="#FFFFFF" transform="rotate(288 12 12)" />
          <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" />
          <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" transform="rotate(72 12 12)" />
          <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" transform="rotate(144 12 12)" />
          <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" transform="rotate(216 12 12)" />
          <line x1="12" y1="6.2" x2="12" y2="10.6" stroke="#a5b4fc" stroke-width="0.7" stroke-linecap="round" opacity="0.8" transform="rotate(288 12 12)" />
          <circle cx="12" cy="12" r="2.1" fill="#818cf8" />
          <circle cx="12" cy="12" r="1.1" fill="#e0e7ff" />
        </svg>
      </div>
      <h3>Hi, I'm Ilayas. What would you like to explore today?</h3>
      <p class="ilayas-tagline">I talk, you understand.</p>
      <p class="ilayas-desc">
        Ask questions across your uploaded documents, synthesize formulas, request diagrams, or generate comparison tables.
      </p>
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
                <span class="citation-pill cursor-pointer" title="${r.section_title ? escapeHtml(r.section_title) : ''} — Click to view PDF on page ${r.page_number || 1}" onclick="openPdfModal('${r.document_id}', '${escapeHtml(r.source_filename)}', ${r.page_number || 1})">
                  📄 <strong>${escapeHtml(r.source_filename)}</strong>
                  ${r.page_number != null ? `· p. ${r.page_number}` : ''}
                  ${r.section_title ? `· ${escapeHtml(r.section_title.substring(0, 30))}` : ''}
                  <span style="color:var(--status-completed-text); font-weight:600;">(${Math.round(r.relevance_score * 100)}%)</span>
                  <span class="material-symbols-outlined text-[12px] opacity-75">open_in_new</span>
                </span>
              `).join('')}
            </div>
          `;
        }

        const fallbackAns = msg.direct_answer || msg.content || "";
        if (fallbackAns) {
          bubbleContent += getChatCopyButtonHtml(fallbackAns);
        }

        msgEl.innerHTML = `
          ${getIlayasAvatarHtml()}
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
  if (!conversationId) {
    startNewChat();
    return;
  }
  activeConversationId = conversationId;
  localStorage.setItem("nalanda_active_conversation_id", conversationId);
  if (activeFolderId) {
    localStorage.setItem(`nalanda_active_conv_${activeFolderId}`, conversationId);
  }
  await loadConversationMessages(conversationId);
}

function startNewChat() {
  activeConversationId = null;
  localStorage.setItem("nalanda_active_conversation_id", "new");
  if (activeFolderId) {
    localStorage.setItem(`nalanda_active_conv_${activeFolderId}`, "new");
  }
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
  const convToDelete = activeConversationId;
  try {
    localStorage.setItem("nalanda_active_conversation_id", "new");
    if (activeFolderId) {
      localStorage.setItem(`nalanda_active_conv_${activeFolderId}`, "new");
    }
    await apiFetch(`/api/conversations/${convToDelete}`, { method: "DELETE" });
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
    modal.style.display = "flex";
    modal.classList.add("active");
  }
}

function closeFigureModal() {
  const modal = document.getElementById("figure-modal");
  if (modal) {
    modal.style.display = "none";
    modal.classList.remove("active");
  }
}

function openPdfModal(documentId, filename, pageNumber = 1) {
  if (!documentId) {
    showToast("Document ID not available", "error");
    return;
  }
  const modal = document.getElementById("pdf-modal");
  const iframe = document.getElementById("pdf-modal-iframe");
  const title = document.getElementById("pdf-modal-title");
  const badge = document.getElementById("pdf-modal-page-badge");
  const newTabBtn = document.getElementById("pdf-modal-open-new-tab");

  if (!modal || !iframe) return;

  const authQuery = token ? `?token=${encodeURIComponent(token)}` : "";
  const pageFragment = pageNumber ? `#page=${pageNumber}` : "";
  const pdfUrl = `/api/documents/${documentId}/file${authQuery}${pageFragment}`;

  if (title) title.textContent = filename || "Document Preview";
  if (badge) {
    if (pageNumber && pageNumber > 1) {
      badge.textContent = `Page ${pageNumber}`;
      badge.style.display = "inline-flex";
    } else {
      badge.style.display = "none";
    }
  }
  if (newTabBtn) {
    newTabBtn.href = pdfUrl;
  }

  iframe.src = pdfUrl;
  modal.style.display = "flex";
  modal.classList.add("active");
}

function closePdfModal() {
  const modal = document.getElementById("pdf-modal");
  const iframe = document.getElementById("pdf-modal-iframe");
  if (modal) {
    modal.style.display = "none";
    modal.classList.remove("active");
  }
  if (iframe) {
    iframe.src = "";
  }
}

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closePdfModal();
    closeFigureModal();
  }
});

async function selectFolder(folderId) {
  if (searchController) {
    searchController.abort();
    searchController = null;
    const submitBtn = document.getElementById("rag-submit");
    if (submitBtn) submitBtn.disabled = false;
  }
  activeFolderId = folderId;
  localStorage.setItem("nalanda_active_folder_id", folderId);
  await loadDocuments(folderId);
  renderFolders();
  renderActiveView();
  await loadFolderConversations(folderId);

}

const CHAT_COPY_SVG = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
const CHAT_CHECK_SVG = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;

function getChatCopyButtonHtml(rawText) {
  const encoded = encodeURIComponent(rawText || "");
  return `
    <div class="chat-message-footer">
      <button class="chat-copy-icon-btn" data-copy-text="${encoded}" onclick="copyChatMessage(this, event)" title="Copy answer to clipboard">
        ${CHAT_COPY_SVG}
        <span>Copy</span>
      </button>
    </div>
  `;
}

function copyToClipboard(text) {
  if (!text) return Promise.reject(new Error("No text to copy"));
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  return new Promise((resolve, reject) => {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    textarea.style.top = "-9999px";
    textarea.setAttribute("readonly", "");
    document.body.appendChild(textarea);
    textarea.select();
    try {
      const successful = document.execCommand("copy");
      document.body.removeChild(textarea);
      if (successful) resolve();
      else reject(new Error("document.execCommand failed"));
    } catch (err) {
      document.body.removeChild(textarea);
      reject(err);
    }
  });
}

async function copyChatMessage(btn, event) {
  if (event) {
    event.stopPropagation();
    event.preventDefault();
  }
  if (!btn) return;

  let textToCopy = "";

  if (btn.dataset.copyText) {
    try {
      textToCopy = decodeURIComponent(btn.dataset.copyText);
    } catch (_) {
      textToCopy = btn.dataset.copyText;
    }
  }

  if (!textToCopy) {
    const messageEl = btn.closest(".chat-message");
    if (messageEl) {
      const bodyEl = messageEl.querySelector(".direct-answer-body");
      if (bodyEl) {
        textToCopy = bodyEl.innerText.trim();
      } else {
        const bubble = messageEl.querySelector(".chat-bubble");
        if (bubble) {
          const clone = bubble.cloneNode(true);
          clone.querySelectorAll(".citations-group, .diagram-card, .table-card, button, .typewriter-skip-hint").forEach(el => el.remove());
          textToCopy = clone.innerText.trim();
        }
      }
    }
  }

  if (!textToCopy) {
    showToast("No text available to copy.", "error");
    return;
  }

  try {
    await copyToClipboard(textToCopy);
    showToast("📋 Copied answer to clipboard!", "success");

    btn.classList.add("copied");
    btn.innerHTML = `
      ${CHAT_CHECK_SVG}
      <span style="color:#059669; font-weight:600;">Copied!</span>
    `;

    setTimeout(() => {
      btn.classList.remove("copied");
      btn.innerHTML = `
        ${CHAT_COPY_SVG}
        <span>Copy</span>
      `;
    }, 2000);
  } catch (err) {
    console.error("Clipboard error:", err);
    showToast("Failed to copy to clipboard.", "error");
  }
}

function formatInlineMarkdown(str) {
  let esc = escapeHtml(str);
  esc = esc.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  esc = esc.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08); padding:1px 5px; border-radius:4px; font-family:monospace; font-size:0.9em;">$1</code>');

  // Math & Typography enhancements: superscripts, subscripts, symbols
  esc = esc.replace(/\\hat\{([a-zA-Z0-9]+)\}/g, '$1&#770;');
  esc = esc.replace(/\\mathbf\{([^\}]+)\}/g, '<strong>$1</strong>');
  esc = esc.replace(/\\text\{([^\}]+)\}/g, '$1');
  esc = esc.replace(/\\times/g, '×');
  esc = esc.replace(/\\cdot/g, '·');
  esc = esc.replace(/\\to/g, '→');
  esc = esc.replace(/\\frac\{([^\}]+)\}\{([^\}]+)\}/g, '($1)/($2)');
  esc = esc.replace(/\\sum_\{([^}]+)\}\^\{([^}]+)\}/g, '∑<sub style="font-size:0.75em; vertical-align:sub;">$1</sub><sup style="font-size:0.75em; vertical-align:super;">$2</sup>');
  esc = esc.replace(/\\sum/g, '∑');

  // Caret superscripts: x^2, (a-b)^2, 10^6, ^{2}, ^2
  esc = esc.replace(/\^\{([0-9a-zA-Z\+\-\(\)\.]+)\}/g, '<sup style="font-size:0.75em; vertical-align:super;">$1</sup>');
  esc = esc.replace(/\^([0-9a-zA-Z\+\-\(\)\.]+)\^/g, '<sup style="font-size:0.75em; vertical-align:super;">$1</sup>');
  esc = esc.replace(/([a-zA-Z0-9_\)])\^([0-9a-zA-Z\+\-]+|\([^\)]+\))/g, '$1<sup style="font-size:0.75em; vertical-align:super;">$2</sup>');

  // Subscripts: W_{conv}, x_1, ~1~
  esc = esc.replace(/_\{([0-9a-zA-Z\+\-\(\)\.\_]+)\}/g, '<sub style="font-size:0.75em; vertical-align:sub;">$1</sub>');
  esc = esc.replace(/~([0-9a-zA-Z\+\-\(\)\.]+)~/g, '<sub style="font-size:0.75em; vertical-align:sub;">$1</sub>');
  esc = esc.replace(/([a-zA-Z0-9])_([0-9a-zA-Z]+|\([^\)]+\))/g, '$1<sub style="font-size:0.75em; vertical-align:sub;">$2</sub>');

  // Strip math $ delimiters from inline formulas
  esc = esc.replace(/\$([^\$]+)\$/g, '$1');

  return esc;
}

function formatDirectAnswerHtml(text) {
  if (!text) return "";
  // Strip raw HTML image or layout tags produced by OCR engines
  let cleanText = text
    .replace(/(?:<div[^>]*>\s*)?<img[^>]*src=["\'](?:imgs\/)?img_in_image_box_[^"\']+["\'][^>]*>(?:\s*<\/div>)?/gi, "")
    .replace(/<div[^>]*style=["\'][^"\']*text-align:\s*center[^"\']*["\'][^>]*>[^<]*<\/div>/gi, "")
    .replace(/<div[^>]*>|<\/div>/gi, "")
    .replace(/<img[^>]*>/gi, "");

  // Ensure any accidentally joined inline headers or goals are broken into separate lines, but do not break '1. Goal' or '### Header'
  const normalizedText = cleanText
    .replace(/(?<!^)(?<!\n)(?<!#)\s{2,}(#{1,4}\s+)/g, "\n$1")
    .replace(/(?<!^)(?<!\n)(?<!\d\.)(?<!\d\))\s{2,}(\*{0,2}(?:Goal|Step|Objective|Feature|Phase|Stage)\s+\d+[:.]?\*{0,2})/gi, "\n$1");

  const lines = normalizedText.split("\n").map(l => l.trim()).filter(l => l.length > 0);
  const htmlParts = [];
  let inList = false;

  for (const line of lines) {
    const headingMatch = line.match(/^#{1,4}\s+(.+)$/);
    const combinedStepGoal = line.match(/^(?:\d+[\.\)]\s+)?(?:\*{0,2}(Goal|Objective|Feature|Stage|Phase|Rule|Step)\s+(\d+)[:.]?\*{0,2})\s*(.*)$/i);
    const stepMatch = line.match(/^(\d+[\.\)])\s*(.+)$/);
    const bulletMatch = line.match(/^(?:[-•+]|\*(?!\*))\s*(.+)$/);

    if (headingMatch && !combinedStepGoal) {
      if (inList) {
        htmlParts.push("</ul>");
        inList = false;
      }
      htmlParts.push(`<div class="direct-answer-section-header" style="font-weight:700; margin-top:8px; margin-bottom:4px; color:var(--text-primary);">${formatInlineMarkdown(headingMatch[1])}</div>`);
    } else if (line.startsWith("$$") && line.endsWith("$$")) {
      if (inList) {
        htmlParts.push("</ul>");
        inList = false;
      }
      const rawEq = line.replace(/^\$\$\s*/, "").replace(/\s*\$\$$/, "");
      htmlParts.push(`<div class="direct-answer-equation" style="background:rgba(99,102,241,0.08); border-left:3px solid var(--brand-primary); padding:10px 14px; margin:10px 0; font-family:monospace; font-size:1.0rem; border-radius:4px; overflow-x:auto;">${formatInlineMarkdown(rawEq)}</div>`);
    } else if (line.includes(" → ")) {
      if (inList) {
        htmlParts.push("</ul>");
        inList = false;
      }
      htmlParts.push(`<div class="direct-answer-flow" style="background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25); border-radius:8px; padding:10px 14px; margin:10px 0; font-size:0.95rem; line-height:1.6; color:var(--text-primary);">${formatInlineMarkdown(line)}</div>`);
    } else if (combinedStepGoal) {
      if (inList) {
        htmlParts.push("</ul>");
        inList = false;
      }
      const label = `${combinedStepGoal[1]} ${combinedStepGoal[2]}`;
      let rest = combinedStepGoal[3] || "";
      rest = rest.replace(/^[:.-]\s*/, "").replace(/\*{1,2}$/, "").replace(/^\*{1,2}/, "").trim();
      htmlParts.push(`
        <div class="direct-answer-step" style="margin-top:8px;">
          <span class="direct-answer-step-num" style="min-width:64px; font-weight:700; background:rgba(99,102,241,0.18); color:var(--brand-primary); padding:3px 8px; border-radius:6px; font-size:0.82rem;">${escapeHtml(label)}</span>
          <span class="direct-answer-step-text" style="font-weight:600; color:var(--text-primary); margin-left:8px;">${formatInlineMarkdown(rest)}</span>
        </div>
      `);
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
      if (line.length < 80 && !/[.:;]$/.test(line) && /^[A-Z]/.test(line) && !line.startsWith("Based on") && !line.startsWith("According to") && !line.startsWith("Example:")) {
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
  if (!containerEl || !phrases || phrases.length === 0) return () => { };

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
        localStorage.setItem("nalanda_active_conversation_id", activeConversationId);
        localStorage.setItem(`nalanda_active_conv_${folderId}`, activeConversationId);
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
    let data;
    const responseText = await response.text();
    try {
      data = JSON.parse(responseText);
    } catch {
      data = { detail: responseText || "Search failed." };
    }
    if (!response.ok) throw new Error(data.detail || `Server error (${response.status})`);
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
        .catch(() => { });
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
              <span class="citation-pill cursor-pointer" title="${r.section_title ? escapeHtml(r.section_title) : ''} — Click to view PDF on page ${r.page_number || 1}" onclick="openPdfModal('${r.document_id}', '${escapeHtml(r.source_filename)}', ${r.page_number || 1})">
                📄 <strong>${escapeHtml(r.source_filename)}</strong>
                ${r.page_number != null ? `· p. ${r.page_number}` : ''}
                ${r.section_title ? `· ${escapeHtml(r.section_title.substring(0, 30))}` : ''}
                <span style="color:var(--status-completed-text); font-weight:600;">(${Math.round(r.relevance_score * 100)}%)</span>
                <span class="material-symbols-outlined text-[12px] opacity-75">open_in_new</span>
              </span>
            `).join('')}
          </div>
        `;
      }
    }

    const fullAnswer = data.direct_answer || "";
    if (fullAnswer) {
      bubbleContent += getChatCopyButtonHtml(fullAnswer);
    }

    assistantMsgEl.innerHTML = `
      ${getIlayasAvatarHtml()}
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
        ${getIlayasAvatarHtml()}
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
  const activeDoc = documents.find((d) => d.id === activeDocumentId);

  // Top Navbar Breadcrumb
  const folderNameEl = document.getElementById("topbar-folder-name");
  const docNameEl = document.getElementById("topbar-doc-name");
  const sepEl = document.getElementById("topbar-sep");
  const countEl = document.getElementById("topbar-doc-count");

  if (activeFolder) {
    if (folderNameEl) folderNameEl.textContent = activeFolder.name;
    if (activeDoc) {
      if (docNameEl) {
        docNameEl.textContent = activeDoc.original_filename;
        docNameEl.style.display = "inline";
      }
      if (sepEl) sepEl.style.display = "inline";
    } else {
      if (docNameEl) docNameEl.style.display = "none";
      if (sepEl) sepEl.style.display = "none";
    }
    if (countEl) countEl.textContent = `${documents.length} Document${documents.length === 1 ? '' : 's'} in Scope`;

    // Home Scope Banner
    const homeFolderName = document.getElementById("home-scope-folder-name");
    const homeDocCount = document.getElementById("home-scope-doc-count");
    if (homeFolderName) homeFolderName.textContent = activeFolder.name;
    if (homeDocCount) homeDocCount.textContent = `(${documents.length} document${documents.length === 1 ? '' : 's'} active)`;

    // Bento Cards
    const bentoNotesTitle = document.getElementById("bento-notes-title");
    const bentoNotesDesc = document.getElementById("bento-notes-desc");
    if (bentoNotesTitle) bentoNotesTitle.textContent = `Study Guide: ${activeFolder.name}`;
    if (bentoNotesDesc) bentoNotesDesc.textContent = `Comprehensive notes synthesized across ${documents.length} lecture unit(s).`;

    const bentoQuizTitle = document.getElementById("bento-quiz-title");
    if (bentoQuizTitle) bentoQuizTitle.textContent = `Practice Quiz: ${activeFolder.name}`;

    const bentoGraphTitle = document.getElementById("bento-graph-title");
    if (bentoGraphTitle) bentoGraphTitle.textContent = `Knowledge Graph: ${activeFolder.name}`;
    const bentoGraphNodesCount = document.getElementById("bento-graph-nodes-count");
    if (bentoGraphNodesCount) bentoGraphNodesCount.textContent = `${documents.length} document${documents.length === 1 ? '' : 's'}`;

    // Chat Scope Label
    const chatScopeLabel = document.getElementById("rag-scope-label");
    if (chatScopeLabel) chatScopeLabel.textContent = `Personal AI Learning Assistant · ${activeFolder.name}`;
  } else {
    if (folderNameEl) folderNameEl.textContent = "Select a Folder";
    if (docNameEl) docNameEl.style.display = "none";
    if (sepEl) sepEl.style.display = "none";
    if (countEl) countEl.textContent = "0 Documents in Scope";
    const chatScopeLabel = document.getElementById("rag-scope-label");
    if (chatScopeLabel) chatScopeLabel.textContent = "Personal AI Learning Assistant";
  }

  // User Greeting
  const greetingEl = document.getElementById("home-greeting-heading");
  if (greetingEl) {
    const name = currentUser ? currentUser.username : "Scholar";
    greetingEl.textContent = `What are we studying today, ${name}?`;
  }

  renderRecentDocuments();
  renderSourceDrawerList();
}

function getDocFileInfo(doc) {
  const filename = (doc && (doc.original_filename || doc.name)) || "document";
  const extMatch = filename.match(/\.([a-zA-Z0-9]+)$/);
  const ext = extMatch ? extMatch[1].toUpperCase() : (doc && doc.file_type ? doc.file_type.toUpperCase() : "FILE");
  const extLower = ext.toLowerCase();

  let icon = "description";
  let iconColor = "text-primary";
  let badgeClass = "bg-primary-container text-on-primary-container border-primary/20";

  if (extLower === "pdf" || (doc && doc.file_type && doc.file_type.toLowerCase() === "pdf")) {
    icon = "picture_as_pdf";
    iconColor = "text-error";
    badgeClass = "bg-red-500/10 text-red-500 border-red-500/30";
  } else if (extLower === "pptx" || (doc && doc.file_type && doc.file_type.toLowerCase() === "pptx")) {
    icon = "slideshow";
    iconColor = "text-amber-500";
    badgeClass = "bg-amber-500/10 text-amber-500 border-amber-500/30";
  } else if (["png", "jpg", "jpeg", "heic", "heif"].includes(extLower) || (doc && doc.file_type && doc.file_type.toLowerCase() === "image")) {
    icon = "image";
    iconColor = "text-emerald-500";
    badgeClass = "bg-emerald-500/10 text-emerald-500 border-emerald-500/30";
  }

  return { ext, extLower, icon, iconColor, badgeClass };
}

function renderRecentDocuments() {
  const container = document.getElementById("home-recent-docs-list");
  if (!container) return;

  if (!documents || documents.length === 0) {
    container.innerHTML = `
      <div class="p-6 text-center text-outline text-body-sm">
        No documents uploaded yet in this folder. Click "+ Upload Material" to begin.
      </div>
    `;
    return;
  }

  const activeFolder = folders.find(f => f.id === activeFolderId);
  const folderName = activeFolder ? activeFolder.name : "Workspace";

  container.innerHTML = documents.map(doc => {
    const { ext, extLower, icon, iconColor, badgeClass } = getDocFileInfo(doc);
    const isDocActive = doc.id === activeDocumentId;

    let statusPill = "";
    if (doc.status === "completed") {
      statusPill = `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-label-sm text-tertiary bg-tertiary-fixed/40 font-medium"><span class="material-symbols-outlined text-[14px]">check</span> Indexed</span>`;
    } else if (doc.status === "processing") {
      statusPill = `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-label-sm text-primary bg-primary-fixed font-medium"><span class="material-symbols-outlined text-[14px] animate-spin">refresh</span> Processing</span>`;
    } else if (doc.status === "failed") {
      statusPill = `<button class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-label-sm text-error bg-error-container font-medium hover:underline" onclick="handleRetry('${doc.id}')">Retry</button>`;
    } else {
      statusPill = `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-label-sm text-outline bg-surface-container font-medium">Pending</span>`;
    }

    let viewButtonLabel = "View File";
    let viewButtonIcon = "visibility";
    if (extLower === "pdf") {
      viewButtonLabel = "View PDF";
    } else if (["png", "jpg", "jpeg"].includes(extLower)) {
      viewButtonLabel = "View Image";
    }

    return `
      <div class="flex items-center justify-between p-3.5 hover:bg-surface-container-low transition-colors cursor-pointer" onclick="selectDocument('${doc.id}', event)">
        <div class="flex items-center gap-3 truncate">
          <div class="w-9 h-9 rounded-lg bg-surface-container-low flex items-center justify-center ${iconColor} shrink-0">
            <span class="material-symbols-outlined">${icon}</span>
          </div>
          <div class="flex flex-col truncate">
            <div class="flex items-center gap-2 truncate">
              <span class="text-body-md font-medium text-on-surface truncate ${isDocActive ? 'text-primary' : ''}">${escapeHtml(doc.original_filename)}</span>
              <span class="px-1.5 py-0.2 rounded text-[10px] font-bold uppercase ${badgeClass} border shrink-0">${ext}</span>
            </div>
            <span class="text-label-sm text-outline">${folderName} • ${formatBytes(doc.file_size_bytes)} • ${formatDate(doc.created_at)}</span>
          </div>
        </div>
        <div class="flex items-center gap-2.5 shrink-0">
          ${statusPill}
          <button class="px-2.5 py-1 rounded-lg border border-outline-variant text-label-sm font-medium text-on-surface hover:text-primary hover:border-primary/50 flex items-center gap-1 transition-colors" title="View Document" onclick="event.stopPropagation(); openPdfModal('${doc.id}', '${escapeHtml(doc.original_filename)}')">
            <span class="material-symbols-outlined text-[15px]">${viewButtonIcon}</span>
            <span>${viewButtonLabel}</span>
          </button>
          <button class="px-2.5 py-1 rounded-lg border border-outline-variant text-label-sm font-medium text-outline hover:text-error hover:border-error/50 hover:bg-error-container/20 flex items-center gap-1 transition-colors" title="Delete Document" onclick="event.stopPropagation(); handleDeleteDocument('${doc.id}', '${escapeHtml(doc.original_filename).replace(/'/g, "\\'")}')">
            <span class="material-symbols-outlined text-[15px]">delete</span>
            <span>Delete</span>
          </button>
          <button class="w-8 h-8 rounded-lg flex items-center justify-center text-outline hover:text-on-surface hover:bg-surface-container transition-colors" title="Select in context" onclick="selectDocument('${doc.id}', event)">
            <span class="material-symbols-outlined">chevron_right</span>
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function renderSourceDrawerList() {
  const container = document.getElementById("nlm-sources-list");
  const countLabel = document.getElementById("nlm-source-count-label");
  if (!container) return;

  if (countLabel) {
    countLabel.textContent = `${documents.length} source${documents.length === 1 ? '' : 's'}`;
  }

  if (!documents || documents.length === 0) {
    container.innerHTML = `
      <div class="text-center text-outline text-body-sm py-8">
        No documents in this folder.
      </div>
    `;
    return;
  }

  container.innerHTML = documents.map(doc => {
    const { ext, extLower, icon, iconColor, badgeClass } = getDocFileInfo(doc);
    const isDocActive = doc.id === activeDocumentId;

    let statusPill = "";
    if (doc.status === "completed") {
      statusPill = `<span class="inline-flex items-center gap-1 text-[12px] text-tertiary font-medium">✓ Indexed</span>`;
    } else if (doc.status === "processing") {
      statusPill = `<span class="inline-flex items-center gap-1 text-[12px] text-primary font-medium">⚡ Processing</span>`;
    } else if (doc.status === "failed") {
      statusPill = `<button class="text-[12px] text-error font-medium hover:underline" onclick="handleRetry('${doc.id}')">⚠️ Retry</button>`;
    } else {
      statusPill = `<span class="text-[12px] text-outline">Pending</span>`;
    }

    const summaryBtn = doc.ingestion_summary ? `
      <button class="p-1 rounded-lg text-outline hover:text-primary hover:bg-surface-container transition-colors" title="Knowledge Integration Summary" onclick="event.stopPropagation(); openIngestionSummaryModal(${escapeHtml(JSON.stringify(doc.ingestion_summary))}, '${escapeHtml(doc.original_filename)}')">
        <span class="material-symbols-outlined text-[17px]">auto_stories</span>
      </button>
    ` : '';

    return `
      <div class="p-2.5 rounded-xl border border-outline-variant bg-surface-container-low hover:border-primary/40 transition-all flex items-center justify-between gap-2">
        <div class="flex items-center gap-2.5 truncate cursor-pointer" onclick="selectDocument('${doc.id}', event)">
          <span class="material-symbols-outlined text-[18px] ${iconColor} shrink-0">${icon}</span>
          <div class="flex flex-col truncate">
            <div class="flex items-center gap-1.5 truncate">
              <span class="text-body-sm font-medium text-on-surface truncate ${isDocActive ? 'text-primary' : ''}">${escapeHtml(doc.original_filename)}</span>
              <span class="px-1 py-0.2 rounded text-[9px] font-bold uppercase ${badgeClass} border shrink-0">${ext}</span>
            </div>
            <span class="text-label-sm text-outline">${formatBytes(doc.file_size_bytes)}</span>
          </div>
        </div>
        <div class="flex items-center gap-1.5 shrink-0">
          ${statusPill}
          ${summaryBtn}
          <button class="p-1 rounded-lg text-outline hover:text-primary hover:bg-surface-container transition-colors" title="View Document" onclick="event.stopPropagation(); openPdfModal('${doc.id}', '${escapeHtml(doc.original_filename)}')">
            <span class="material-symbols-outlined text-[17px]">visibility</span>
          </button>
          <button class="p-1 rounded-lg text-outline hover:text-error hover:bg-surface-container transition-colors" title="Delete Document" onclick="event.stopPropagation(); handleDeleteDocument('${doc.id}', '${escapeHtml(doc.original_filename)}')">
            <span class="material-symbols-outlined text-[17px]">delete</span>
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function renderDocumentsTable() {
  renderRecentDocuments();
  renderSourceDrawerList();
}

let knownProcessedDocIds = new Set();

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
            const updatedDocs = await res.json();
            // Check for newly completed documents to present knowledge integration summary
            updatedDocs.forEach((d) => {
              if (d.status === "completed" && !knownProcessedDocIds.has(d.id)) {
                knownProcessedDocIds.add(d.id);
                if (d.ingestion_summary) {
                  openIngestionSummaryModal(d.ingestion_summary, d.original_filename);
                }
              }
            });

            documents = updatedDocs;
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

function openIngestionSummaryModal(summary, filename) {
  const modal = document.getElementById("ingestion-summary-modal");
  if (!modal || !summary) return;

  const nameEl = document.getElementById("ingestion-modal-filename");
  const matchedEl = document.getElementById("summary-matched-count");
  const enrichedEl = document.getElementById("summary-enriched-count");
  const createdEl = document.getElementById("summary-created-count");
  const skippedEl = document.getElementById("summary-skipped-count");
  const conflictsBox = document.getElementById("ingestion-modal-conflicts-box");
  const conflictsMsg = document.getElementById("summary-conflicts-msg");
  const msgEl = document.getElementById("ingestion-modal-message");

  if (nameEl) nameEl.textContent = filename || "Uploaded Document";
  if (matchedEl) matchedEl.textContent = summary.items_matched || 0;
  if (enrichedEl) enrichedEl.textContent = summary.concepts_enriched || 0;
  if (createdEl) createdEl.textContent = summary.concepts_created || 0;
  if (skippedEl) skippedEl.textContent = summary.duplicates_skipped || 0;

  if (conflictsBox && conflictsMsg) {
    if (summary.conflicts_flagged && summary.conflicts_flagged > 0) {
      conflictsBox.classList.remove("hidden");
      conflictsMsg.textContent = `${summary.conflicts_flagged} potentially conflicting statement(s) flagged across documents.`;
    } else {
      conflictsBox.classList.add("hidden");
    }
  }

  if (msgEl) {
    msgEl.textContent = summary.summary_message || "Upload processed and integrated into knowledge base.";
  }

  modal.style.display = "flex";
}

function closeIngestionSummaryModal() {
  const modal = document.getElementById("ingestion-summary-modal");
  if (modal) modal.style.display = "none";
}

async function handleDeleteDocument(documentId, filename) {
  if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
    return;
  }
  try {
    showToast(`Deleting "${filename}"...`);
    const res = await apiFetch(`/api/documents/${documentId}`, {
      method: "DELETE",
    });
    if (res.ok) {
      showToast(`Document "${filename}" deleted successfully.`);
      if (activeDocumentId === documentId) {
        activeDocumentId = null;
      }
      await loadDocuments(activeFolderId);
      await loadFolders();
      renderFolders();
      renderActiveView();
    } else {
      let err;
      try {
        err = await res.json();
      } catch {
        err = { detail: await res.text() };
      }
      showToast(err.detail || "Failed to delete document", "error");
    }
  } catch (err) {
    showToast("Error deleting document", "error");
  }
}

// Supported upload formats
const SUPPORTED_UPLOAD_EXTENSIONS = [".pdf", ".pptx", ".heic", ".heif", ".png", ".jpg", ".jpeg"];
const MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB

// State for tracked uploads (Requirements 4, 5, 6)
let uploadQueue = [];
let uploadWidgetCollapsed = false;

function toggleUploadWidgetCollapse() {
  uploadWidgetCollapsed = !uploadWidgetCollapsed;
  const list = document.getElementById("upload-widget-file-list");
  const icon = document.getElementById("upload-widget-collapse-icon");
  if (list) {
    list.style.display = uploadWidgetCollapsed ? "none" : "flex";
  }
  if (icon) {
    icon.textContent = uploadWidgetCollapsed ? "expand_less" : "expand_more";
  }
}

function closeUploadWidget() {
  const widget = document.getElementById("upload-status-widget");
  if (widget) widget.style.display = "none";
}

function renderUploadQueueUI() {
  const widget = document.getElementById("upload-status-widget");
  const widgetList = document.getElementById("upload-widget-file-list");
  const widgetTitle = document.getElementById("upload-widget-title");
  const drawerQueue = document.getElementById("drawer-upload-queue");

  if (!uploadQueue || uploadQueue.length === 0) {
    if (widget) widget.style.display = "none";
    if (drawerQueue) drawerQueue.style.display = "none";
    return;
  }

  const total = uploadQueue.length;
  const inProgress = uploadQueue.filter(u => u.status === "uploading").length;
  const failed = uploadQueue.filter(u => u.status === "failed").length;
  const completed = uploadQueue.filter(u => u.status === "uploaded").length;

  if (widgetTitle) {
    if (inProgress > 0) {
      widgetTitle.textContent = `Uploading ${inProgress} of ${total} file${total === 1 ? '' : 's'}...`;
    } else if (failed > 0 && completed === 0) {
      widgetTitle.textContent = `Upload failed (${failed} file${failed === 1 ? '' : 's'})`;
    } else if (failed > 0) {
      widgetTitle.textContent = `Upload finished (${completed} ok, ${failed} failed)`;
    } else {
      widgetTitle.textContent = `Uploaded ${completed} file${completed === 1 ? '' : 's'}`;
    }
  }

  const itemsHtml = uploadQueue.slice().reverse().map(item => {
    const { ext, icon, iconColor, badgeClass } = getDocFileInfo({ original_filename: item.name });
    
    let statusBadge = "";
    if (item.status === "uploading") {
      statusBadge = `<span class="inline-flex items-center gap-1 text-[11px] font-medium text-primary"><span class="material-symbols-outlined text-[14px] animate-spin">refresh</span> Uploading</span>`;
    } else if (item.status === "uploaded") {
      statusBadge = `<span class="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-500"><span class="material-symbols-outlined text-[14px]">check_circle</span> Uploaded</span>`;
    } else {
      statusBadge = `<span class="inline-flex items-center gap-1 text-[11px] font-medium text-error"><span class="material-symbols-outlined text-[14px]">error</span> Failed</span>`;
    }

    const errorBlock = item.error ? `
      <div class="mt-1 text-[11px] text-error bg-error-container/20 border border-error/20 rounded px-2 py-1 leading-tight break-words">
        ${escapeHtml(item.error)}
      </div>
    ` : "";

    return `
      <div class="p-2.5 rounded-xl border border-outline-variant bg-surface-container-low flex flex-col gap-1 transition-all">
        <div class="flex items-center justify-between gap-2">
          <div class="flex items-center gap-2 truncate min-w-0">
            <span class="material-symbols-outlined text-[18px] ${iconColor} shrink-0">${icon}</span>
            <div class="flex flex-col truncate">
              <span class="text-body-sm font-medium text-on-surface truncate" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
              <div class="flex items-center gap-1.5 text-label-xs text-outline">
                <span class="px-1 py-0.2 rounded text-[9px] font-bold uppercase ${badgeClass} border">${ext}</span>
                <span>${item.formattedSize}</span>
              </div>
            </div>
          </div>
          <div class="shrink-0">
            ${statusBadge}
          </div>
        </div>
        ${errorBlock}
      </div>
    `;
  }).join("");

  if (widgetList) {
    widgetList.innerHTML = itemsHtml;
  }
  if (widget) {
    widget.style.display = "flex";
  }

  if (drawerQueue) {
    drawerQueue.innerHTML = `
      <div class="text-label-xs text-outline font-semibold uppercase tracking-wider mb-1">Recent Uploads</div>
      ${itemsHtml}
    `;
    drawerQueue.style.display = "flex";
  }
}

// Upload Handling
function triggerFileInput() {
  if (!activeFolderId) {
    showToast("Please select a folder first", "error");
    return;
  }
  const input = document.getElementById("file-input");
  if (input) {
    input.value = "";
    input.click();
  }
}

async function handleMultipleFilesUpload(fileList) {
  if (!fileList || fileList.length === 0) return;

  if (!activeFolderId) {
    showToast("Please select a knowledge folder first", "error");
    return;
  }

  const files = Array.from(fileList);

  for (const file of files) {
    const extMatch = file.name.match(/\.([a-zA-Z0-9]+)$/);
    const ext = extMatch ? `.${extMatch[1].toLowerCase()}` : "";
    const formattedSize = formatBytes(file.size);

    const queueItem = {
      id: "upl_" + Date.now() + "_" + Math.random().toString(36).substring(2, 7),
      file: file,
      name: file.name,
      ext: ext,
      size: file.size,
      formattedSize: formattedSize,
      status: "uploading",
      error: null,
      timestamp: Date.now(),
    };

    // Client-side format validation
    if (!ext || !SUPPORTED_UPLOAD_EXTENSIONS.includes(ext)) {
      const allowedStr = SUPPORTED_UPLOAD_EXTENSIONS.join(", ");
      queueItem.status = "failed";
      queueItem.error = `Unsupported format "${ext || 'none'}". Supported formats: ${allowedStr}`;
      uploadQueue.push(queueItem);
      renderUploadQueueUI();
      showToast(queueItem.error, "error");
      continue;
    }

    // Client-side empty file validation
    if (file.size <= 0) {
      queueItem.status = "failed";
      queueItem.error = "File is empty (0 bytes).";
      uploadQueue.push(queueItem);
      renderUploadQueueUI();
      showToast(`"${file.name}" is empty (0 bytes)`, "error");
      continue;
    }

    // Client-side file size validation
    if (file.size > MAX_UPLOAD_SIZE_BYTES) {
      queueItem.status = "failed";
      queueItem.error = `File size exceeds the 50 MB limit (${formattedSize}).`;
      uploadQueue.push(queueItem);
      renderUploadQueueUI();
      showToast(`"${file.name}" exceeds 50 MB limit`, "error");
      continue;
    }

    uploadQueue.push(queueItem);
    renderUploadQueueUI();

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await apiFetch(`/api/folders/${activeFolderId}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        queueItem.status = "uploaded";
        queueItem.error = null;
        showToast(`Uploaded "${file.name}".`);
      } else {
        const err = await res.json().catch(() => ({}));
        queueItem.status = "failed";
        queueItem.error = err.detail || `Upload failed with status ${res.status}`;
        showToast(queueItem.error, "error");
      }
    } catch (err) {
      queueItem.status = "failed";
      queueItem.error = err.message || "Network error during file upload.";
      showToast(`Error uploading "${file.name}"`, "error");
    }

    renderUploadQueueUI();
  }

  if (activeFolderId) {
    await loadDocuments(activeFolderId);
    await loadFolders();
  }
}

async function handleFileUpload(file) {
  if (file) {
    await handleMultipleFilesUpload([file]);
  }
}

function setupUploadDropzones() {
  const dropzone = document.getElementById("upload-dropzone");
  const recentDocsContainer = document.getElementById("home-recent-docs-list");

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
    window.addEventListener(eventName, preventDefaults, false);
  });

  if (dropzone) {
    ["dragenter", "dragover"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        preventDefaults(e);
        dropzone.classList.add("dropzone-dragover");
        const subtitle = document.getElementById("upload-dropzone-subtitle");
        if (subtitle) subtitle.textContent = "Drop files here to upload";
      }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        preventDefaults(e);
        dropzone.classList.remove("dropzone-dragover");
        const subtitle = document.getElementById("upload-dropzone-subtitle");
        if (subtitle) subtitle.textContent = "PDF, PPTX, HEIC, HEIF, PNG, JPG (up to 50MB)";
      }, false);
    });

    dropzone.addEventListener("drop", (e) => {
      preventDefaults(e);
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleMultipleFilesUpload(dt.files);
      }
    }, false);
  }

  if (recentDocsContainer) {
    ["dragenter", "dragover"].forEach(eventName => {
      recentDocsContainer.addEventListener(eventName, (e) => {
        preventDefaults(e);
        recentDocsContainer.classList.add("dropzone-dragover");
      }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
      recentDocsContainer.addEventListener(eventName, (e) => {
        preventDefaults(e);
        recentDocsContainer.classList.remove("dropzone-dragover");
      }, false);
    });

    recentDocsContainer.addEventListener("drop", (e) => {
      preventDefaults(e);
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleMultipleFilesUpload(dt.files);
      }
    }, false);
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

document.addEventListener("DOMContentLoaded", () => {
  checkAuth();

  const ragForm = document.getElementById("rag-form");
  if (ragForm) ragForm.addEventListener("submit", handleSemanticSearch);

  setupUploadDropzones();

  // Restore exact page and route based on current URL to preserve page on reload
  const currentRoute = getCurrentRoute();
  switchView(currentRoute, false);
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
  switchView("quiz");
  initQuizUnitSelector();
  loadPracticeQuiz();
}

function closePracticeQuizModal() {
  switchView("home");
}

function initQuizUnitSelector() {
  const completedDocs = (documents || []).filter(d => d.status === "completed");
  const selectorBar = document.getElementById("quiz-unit-selector-bar");
  if (!selectorBar) return;

  if (completedDocs.length === 0) {
    selectorBar.style.display = "none";
    return;
  }

  // Restore saved quiz units for this folder if valid
  const validIds = new Set(completedDocs.map(d => d.id));
  let restored = false;
  try {
    const rawSaved = localStorage.getItem(`nalanda_quiz_units_${activeFolderId}`);
    if (rawSaved) {
      const parsed = JSON.parse(rawSaved);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const validSaved = parsed.filter(id => validIds.has(id));
        if (validSaved.length > 0) {
          selectedQuizDocIds = new Set(validSaved);
          restored = true;
        }
      }
    }
  } catch (e) { }

  if (!restored) {
    const hasValid = Array.from(selectedQuizDocIds).some(id => validIds.has(id));
    if (!hasValid) {
      selectedQuizDocIds = new Set(validIds);
    } else {
      selectedQuizDocIds = new Set(Array.from(selectedQuizDocIds).filter(id => validIds.has(id)));
    }
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
  if (activeFolderId) {
    localStorage.setItem(`nalanda_quiz_units_${activeFolderId}`, JSON.stringify(Array.from(selectedQuizDocIds)));
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
  if (activeFolderId) {
    localStorage.setItem(`nalanda_quiz_units_${activeFolderId}`, JSON.stringify(Array.from(selectedQuizDocIds)));
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

  const q = currentQuizQuestions.find(item => item.id === questionId);
  const cleanOption = (text) => (text || "").replace(/^[A-D]\)\s*/i, "").trim().toLowerCase();

  const selectedText = q && q.options && q.options[selectedIdx] ? cleanOption(q.options[selectedIdx]) : "";
  const correctAnswerText = q ? cleanOption(q.correct_answer) : "";

  // Resolve true correct index from both question metadata and option text matching
  let resolvedCorrectIdx = Number(correctIdx);
  if (q && q.options && correctAnswerText) {
    const textMatchedIdx = q.options.findIndex(opt => cleanOption(opt) === correctAnswerText);
    if (textMatchedIdx !== -1) {
      resolvedCorrectIdx = textMatchedIdx;
    }
  }

  // Evaluate correctness: index match, exact text match, or singular/plural equivalence
  const isIndexMatch = Number(selectedIdx) === resolvedCorrectIdx;
  const isTextMatch = Boolean(selectedText && correctAnswerText && (
    selectedText === correctAnswerText ||
    selectedText === correctAnswerText + "s" ||
    selectedText + "s" === correctAnswerText ||
    selectedText === correctAnswerText + "es" ||
    selectedText + "es" === correctAnswerText
  ));
  const isCorrect = isIndexMatch || isTextMatch;

  quizAnswersState[questionId] = isCorrect;

  const selectedBtn = document.getElementById(`btn-${questionId}-${selectedIdx}`);
  const correctBtn = document.getElementById(`btn-${questionId}-${resolvedCorrectIdx}`);
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
  switchView("notes");
  initNotesUnitSelector();
  loadNotes();
}

function closeNotesModal() {
  switchView("home");
}

function initNotesUnitSelector() {
  const completedDocs = (documents || []).filter(d => d.status === "completed");
  const selectorBar = document.getElementById("notes-unit-selector-bar");
  if (!selectorBar) return;

  if (completedDocs.length === 0) {
    selectorBar.style.display = "none";
    return;
  }

  // Restore saved notes units for this folder if valid
  const validIds = new Set(completedDocs.map(d => d.id));
  let restored = false;
  try {
    const rawSaved = localStorage.getItem(`nalanda_notes_units_${activeFolderId}`);
    if (rawSaved) {
      const parsed = JSON.parse(rawSaved);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const validSaved = parsed.filter(id => validIds.has(id));
        if (validSaved.length > 0) {
          selectedUnitDocIds = new Set(validSaved);
          restored = true;
        }
      }
    }
  } catch (e) { }

  if (!restored) {
    const hasValid = Array.from(selectedUnitDocIds).some(id => validIds.has(id));
    if (!hasValid) {
      selectedUnitDocIds = new Set(validIds);
    } else {
      selectedUnitDocIds = new Set(Array.from(selectedUnitDocIds).filter(id => validIds.has(id)));
    }
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
  if (activeFolderId) {
    localStorage.setItem(`nalanda_notes_units_${activeFolderId}`, JSON.stringify(Array.from(selectedUnitDocIds)));
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
  if (activeFolderId) {
    localStorage.setItem(`nalanda_notes_units_${activeFolderId}`, JSON.stringify(Array.from(selectedUnitDocIds)));
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
  html = html.replace(/```(?:[a-zA-Z0-9_\-]+)?\r?\n([\s\S]*?)```/g, function (match, code) {
    return '<pre class="notes-code-block"><code>' + code.trim() + '</code></pre>';
  });

  // Headers (#, ##, ###, ####)
  html = html.replace(/^#### (.*?)$/gm, '<h4 class="notes-h4 notes-heading-bold">$1</h4>');
  html = html.replace(/^### (.*?)$/gm, '<h3 class="notes-h3 notes-heading-bold">$1</h3>');
  html = html.replace(/^## (.*?)$/gm, '<h2 class="notes-h2 notes-heading-bold">$1</h2>');
  html = html.replace(/^# (.*?)$/gm, '<h1 class="notes-h1 notes-heading-bold">$1</h1>');

  // Standard Study Note Headings (Definition, Key Points, Working, Advantages, Disadvantages, etc.)
  const stdHeadingRegex = /^(?:Definition|Key Points|Introduction|Working|Advantages|Disadvantages|Applications|Examples|Conclusion|Differences? Between(?: [A-Za-z0-9\s/–—\-]+)?|Important Points|Explanation|Core Components|Syntax|Exam-Relevant Points|Summary|Overview|Working \/ Core Components|Advantages and Limitations|Comparison \/ Reference Table|Syntax \/ Command Example)(?:\s*[:—\-])?$/i;

  html = html.split("\n").map(line => {
    const trimmed = line.trim();
    if (
      trimmed &&
      trimmed.length <= 65 &&
      !trimmed.startsWith("<") &&
      !trimmed.startsWith("#") &&
      !trimmed.startsWith("|") &&
      !trimmed.startsWith("-") &&
      !trimmed.startsWith("*") &&
      !trimmed.startsWith("•") &&
      !trimmed.endsWith(".")
    ) {
      if (stdHeadingRegex.test(trimmed)) {
        const cleanTitle = trimmed.replace(/[:—\-]$/, "").trim();
        return `<h2 class="notes-h2 notes-heading-bold">${cleanTitle}</h2>`;
      }
    }
    return line;
  }).join("\n");

  // Blockquotes (Callouts)
  html = html.replace(/^&gt; (.*?)$/gm, '<blockquote class="notes-blockquote">$1</blockquote>');

  // Bold & Italic & Inline Code
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  html = html.replace(/`([^`]+)`/g, '<code class="notes-inline-code">$1</code>');

  // Math & Typography enhancements: superscripts, subscripts, symbols
  html = html.replace(/\\hat\{([a-zA-Z0-9]+)\}/g, '$1&#770;');
  html = html.replace(/\\mathbf\{([^\}]+)\}/g, '<strong>$1</strong>');
  html = html.replace(/\\text\{([^\}]+)\}/g, '$1');
  html = html.replace(/\\times/g, '×');
  html = html.replace(/\\cdot/g, '·');
  html = html.replace(/\\to/g, '→');
  html = html.replace(/\\frac\{([^\}]+)\}\{([^\}]+)\}/g, '($1)/($2)');
  html = html.replace(/\\sum_\{([^}]+)\}\^\{([^}]+)\}/g, '∑<sub class="notes-sub">$1</sub><sup class="notes-sup">$2</sup>');
  html = html.replace(/\\sum/g, '∑');

  // Caret superscripts: x^2, (a-b)^2, 10^6, ^{2}, ^2
  html = html.replace(/\^\{([0-9a-zA-Z\+\-\(\)\.]+)\}/g, '<sup class="notes-sup">$1</sup>');
  html = html.replace(/\^([0-9a-zA-Z\+\-\(\)\.]+)\^/g, '<sup class="notes-sup">$1</sup>');
  html = html.replace(/([a-zA-Z0-9_\)])\^([0-9a-zA-Z\+\-]+|\([^\)]+\))/g, '$1<sup class="notes-sup">$2</sup>');

  // Subscripts: W_{conv}, x_1, ~1~
  html = html.replace(/_\{([0-9a-zA-Z\+\-\(\)\.\_]+)\}/g, '<sub class="notes-sub">$1</sub>');
  html = html.replace(/~([0-9a-zA-Z\+\-\(\)\.]+)~/g, '<sub class="notes-sub">$1</sub>');
  html = html.replace(/([a-zA-Z0-9])_([0-9a-zA-Z]+|\([^\)]+\))/g, '$1<sub class="notes-sub">$2</sub>');

  // Display equations ($$...$$) in notes
  html = html.replace(/\$\$([\s\S]*?)\$\$/g, function(match, eq) {
    return '<div class="notes-math-equation" style="background:rgba(99,102,241,0.08); border-left:3px solid var(--brand-primary); padding:10px 14px; margin:12px 0; font-family:monospace; font-size:1.0rem; border-radius:4px; overflow-x:auto;">' + eq.trim() + '</div>';
  });
  html = html.replace(/\$([^\$\n]+)\$/g, '$1');

  // Horizontal Rules
  html = html.replace(/^---$/gm, '<hr class="notes-hr">');

  // Markdown Tables (Robust parsing with border styling and duplicate prevention)
  const seenTableFingerprints = new Set();
  html = html.replace(/((?:^[ \t]*\|[^\n]+\|[ \t]*(?:\r?\n|$))+)/gm, function (tableMatch) {
    const rawLines = tableMatch.trim().split("\n").map(l => l.trim()).filter(l => l.length > 0);
    if (rawLines.length < 2) return tableMatch;

    // Fingerprint based on normalized rows to eliminate duplicate rendered tables
    const fingerprint = rawLines.map(l => l.replace(/\s+/g, " ").toLowerCase()).join("\n");
    if (seenTableFingerprints.has(fingerprint)) {
      return "";
    }
    seenTableFingerprints.add(fingerprint);

    let tableHtml = '<div class="notes-table-container"><table class="notes-table">';
    const headerCells = rawLines[0].split("|").slice(1, -1);
    tableHtml += '<thead><tr>' + headerCells.map(c => `<th>${c.trim()}</th>`).join('') + '</tr></thead><tbody>';

    let startRow = 1;
    if (rawLines.length > 1 && /^\|[\s\-:|]+\|$/.test(rawLines[1])) {
      startRow = 2;
    }

    for (let i = startRow; i < rawLines.length; i++) {
      if (/^\|[\s\-:|]+\|$/.test(rawLines[i])) continue;
      const rowCells = rawLines[i].split("|").slice(1, -1);
      if (rowCells.length > 0) {
        tableHtml += '<tr>' + rowCells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
      }
    }
    tableHtml += '</tbody></table></div>';
    return tableHtml;
  });

  // Numbered Lists (with optional indentation)
  html = html.replace(/^(\s*)(\d+)\.\s+(.*?)$/gm, function (m, indent, num, text) {
    const ml = indent.length > 0 ? ' style="margin-left: ' + (indent.length * 10) + 'px;"' : '';
    return '<div class="notes-list-item numbered"' + ml + '><span class="list-num">' + num + '.</span> <span>' + text + '</span></div>';
  });

  // Bullet Lists (with optional indentation and support for - and *)
  html = html.replace(/^(\s*)[-*]\s+(.*?)$/gm, function (m, indent, text) {
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

/* ==========================================================================
   Stitch Workspace Controllers (View Switching, Drawer, Home Prompt)
   ========================================================================== */

function getCurrentRoute() {
  // 1. Check URL pathname (e.g. /chat, /notes, /quiz, /graph, /home)
  const path = window.location.pathname.replace(/^\/+|\/+$/g, "").toLowerCase();
  if (["home", "chat", "notes", "quiz", "graph"].includes(path)) {
    return path;
  }

  // 2. Check hash route (e.g. #/chat, #chat, #/notes, #/graph)
  const hash = window.location.hash.replace(/^#\/?/, "").toLowerCase();
  if (["home", "chat", "notes", "quiz", "graph"].includes(hash)) {
    return hash;
  }

  // 3. Check query parameter ?view=...
  const params = new URLSearchParams(window.location.search);
  const qView = (params.get("view") || "").toLowerCase();
  if (["home", "chat", "notes", "quiz", "graph"].includes(qView)) {
    return qView;
  }

  // 4. Fallback to localStorage saved view only when on root path "/" or "index.html"
  if (path === "" || path === "index.html") {
    const saved = (localStorage.getItem("nalanda_active_view") || "").toLowerCase();
    if (["home", "chat", "notes", "quiz", "graph"].includes(saved)) {
      return saved;
    }
  }

  return "home";
}

function switchView(viewName, updateUrl = true) {
  if (viewName === "graph" || viewName === "readiness") viewName = "home";
  if (!viewName || !["home", "chat", "notes", "quiz"].includes(viewName)) {
    viewName = "home";
  }
  activeView = viewName;
  localStorage.setItem("nalanda_active_view", viewName);

  // Synchronize browser URL to preserve current page and route without reload
  if (window.history) {
    const targetPath = viewName === "home" ? "/" : `/${viewName}`;
    const currentPath = window.location.pathname;
    const cleanCurrent = currentPath.replace(/^\/+|\/+$/g, "").toLowerCase();

    if (updateUrl && window.history.pushState) {
      if (cleanCurrent !== viewName && !(viewName === "home" && cleanCurrent === "")) {
        window.history.pushState({ view: viewName }, "", targetPath + window.location.search);
      }
    } else if (!updateUrl && window.history.replaceState) {
      // Sync URL on initial page load / refresh
      if (cleanCurrent !== viewName && !(viewName === "home" && cleanCurrent === "")) {
        window.history.replaceState({ view: viewName }, "", targetPath + window.location.search);
      }
    }
  }

  // Update navigation buttons
  document.querySelectorAll(".sidebar-nav-item").forEach(btn => {
    btn.classList.remove("active");
  });
  const activeNavBtn = document.getElementById(`nav-btn-${viewName}`);
  if (activeNavBtn) activeNavBtn.classList.add("active");

  // Show selected view and hide others
  const views = ["home", "chat", "notes", "quiz"];
  views.forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) {
      if (v === viewName) {
        el.style.display = (v === "chat") ? "flex" : "block";
      } else {
        el.style.display = "none";
      }
    }
  });

  // Trigger view-specific loads if necessary
  if (viewName === "notes" && activeFolderId) {
    initNotesUnitSelector();
    if (!currentNotesData) {
      loadNotes();
    }
  } else if (viewName === "quiz" && activeFolderId) {
    initQuizUnitSelector();
    if (currentQuizQuestions.length === 0) {
      loadPracticeQuiz();
    }
  }
}

// Listen to browser Back / Forward buttons and Hash changes
window.addEventListener("popstate", () => {
  const route = getCurrentRoute();
  switchView(route, false);
});

window.addEventListener("hashchange", () => {
  const route = getCurrentRoute();
  switchView(route, false);
});

function toggleSourceDrawer(forceState) {
  const drawer = document.getElementById("source-drawer");
  const backdrop = document.getElementById("source-drawer-backdrop");
  if (!drawer) return;

  const isOpen = !drawer.classList.contains("translate-x-full");
  const shouldOpen = forceState !== undefined ? forceState : !isOpen;

  if (shouldOpen) {
    drawer.classList.remove("translate-x-full");
    if (backdrop) backdrop.classList.remove("hidden");
  } else {
    drawer.classList.add("translate-x-full");
    if (backdrop) backdrop.classList.add("hidden");
  }
}

function toggleSidebarCollapse() {
  const sidebar = document.getElementById("sidebar");
  const mainCanvas = document.getElementById("main-canvas");
  if (!sidebar || !mainCanvas) return;

  isSidebarCollapsed = !isSidebarCollapsed;
  if (isSidebarCollapsed) {
    sidebar.classList.add("-translate-x-full");
    mainCanvas.classList.remove("ml-64");
    mainCanvas.classList.add("ml-0");
  } else {
    sidebar.classList.remove("-translate-x-full");
    mainCanvas.classList.add("ml-64");
    mainCanvas.classList.remove("ml-0");
  }
}

function handleHomePromptKey(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleHomePromptSubmit();
  }
}

function handleHomePromptSubmit() {
  const homeInput = document.getElementById("home-prompt-input");
  if (!homeInput) return;
  const text = homeInput.value.trim();
  if (!text) return;

  homeInput.value = "";
  // Always start as a fresh new chat session when initiated from the home page
  startNewChat();
  switchView("chat");
  const chatInput = document.getElementById("rag-query");
  if (chatInput) {
    chatInput.value = text;
  }
  const ragForm = document.getElementById("rag-form");
  if (ragForm) {
    ragForm.dispatchEvent(new Event("submit", { cancelable: true }));
  }
}

function quickCompareDocuments() {
  // Always start as a fresh new chat session when initiated from the home page
  startNewChat();
  switchView("chat");
  const chatInput = document.getElementById("rag-query");
  if (chatInput) {
    chatInput.value = "Compare the key architectures, technologies, and core concepts across the documents in this course.";
  }
  const ragForm = document.getElementById("rag-form");
  if (ragForm) {
    ragForm.dispatchEvent(new Event("submit", { cancelable: true }));
  }
}

function handleShareWorkspace() {
  const activeFolder = folders.find(f => f.id === activeFolderId);
  const name = activeFolder ? activeFolder.name : "Nalanda Workspace";
  if (navigator.clipboard) {
    navigator.clipboard.writeText(window.location.href);
    showToast(`🔗 Link to workspace "${name}" copied to clipboard!`);
  }
}

function toggleFullScreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  } else if (document.exitFullscreen) {
    document.exitFullscreen().catch(() => {});
  }
}

function toggleNewMenu() {
  const menu = document.getElementById("new-dropdown-menu");
  if (menu) menu.classList.toggle("hidden");
}

function closeNewMenu() {
  const menu = document.getElementById("new-dropdown-menu");
  if (menu) menu.classList.add("hidden");
}

document.addEventListener("click", (e) => {
  const newBtn = document.getElementById("new-btn");
  const newMenu = document.getElementById("new-dropdown-menu");
  if (newMenu && !newMenu.classList.contains("hidden")) {
    if (newBtn && !newBtn.contains(e.target) && !newMenu.contains(e.target)) {
      newMenu.classList.add("hidden");
    }
  }
});

function handleSidebarSearch(query) {
  const q = (query || "").trim().toLowerCase();
  const folderItems = document.querySelectorAll(".folder-tree-item");
  folderItems.forEach(item => {
    if (!q) {
      item.style.display = "flex";
      return;
    }
    const text = item.textContent.toLowerCase();
    item.style.display = text.includes(q) ? "flex" : "none";
  });
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeCreateFolderModal();
    closeDeleteFolderModal();
    closeFigureModal();
    closeNewMenu();
  }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    const searchInput = document.getElementById("sidebar-search-input");
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    }
  }
});

function handleFileSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    handleMultipleFilesUpload(files);
    event.target.value = "";
  }
}

function applySuggestedPrompt(text) {
  const input = document.getElementById("rag-query");
  if (input) {
    input.value = text;
    input.focus();
  }
  const ragForm = document.getElementById("rag-form");
  if (ragForm) {
    ragForm.dispatchEvent(new Event("submit", { cancelable: true }));
  }
}

function toggleSelectAllSources(isChecked) {
  const checkboxes = document.querySelectorAll("#nlm-sources-list input[type='checkbox']");
  checkboxes.forEach(cb => {
    cb.checked = isChecked;
  });
}

function toggleSourceCheck(docId, isChecked) {
  // Can be used to filter active context sources
}
