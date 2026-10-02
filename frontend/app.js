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

async function selectFolder(folderId) {
  if (searchController) {
    searchController.abort();
    searchController = null;
    document.getElementById("rag-submit").disabled = false;
  }
  document.getElementById("rag-status").textContent = "Enter a question to find relevant sources.";
  document.getElementById("rag-results").replaceChildren();
  activeFolderId = folderId;
  renderFolders();
  renderActiveView();
  await loadDocuments(folderId);
}

async function handleSemanticSearch(event) {
  event.preventDefault();
  const queryInput = document.getElementById("rag-query");
  const submitButton = document.getElementById("rag-submit");
  const status = document.getElementById("rag-status");
  const results = document.getElementById("rag-results");
  const query = queryInput.value.trim();
  const folderId = activeFolderId;

  if (!query || !folderId) return;

  if (searchController) searchController.abort();
  const controller = new AbortController();
  searchController = controller;
  submitButton.disabled = true;
  status.textContent = "Searching this folder...";
  results.replaceChildren();

  try {
    const response = await apiFetch("/api/search/semantic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, folder_id: folderId, top_k: 5 }),
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Search failed.");
    if (folderId !== activeFolderId) return;

    status.textContent = data.total_results
      ? `${data.total_results} relevant source${data.total_results === 1 ? "" : "s"}`
      : "No relevant sources found in this folder.";

    data.results.forEach((result) => {
      const item = document.createElement("li");
      const heading = document.createElement("div");
      heading.className = "rag-result-heading";

      const filename = document.createElement("strong");
      filename.textContent = result.source_filename;
      heading.appendChild(filename);

      const score = document.createElement("span");
      score.className = "rag-score";
      score.textContent = `${Math.round(result.relevance_score * 100)}% match`;
      heading.appendChild(score);
      item.appendChild(heading);

      const details = [];
      if (result.page_number != null) details.push(`Page ${result.page_number}`);
      if (result.section_title) details.push(result.section_title);
      const detail = document.createElement("p");
      detail.textContent = details.length ? details.join(" · ") : "Source document";
      item.appendChild(detail);
      results.appendChild(item);
    });
  } catch (error) {
    if (error.name !== "AbortError") {
      status.textContent = error.message || "Search is unavailable. Please try again.";
    }
  } finally {
    if (searchController === controller) {
      searchController = null;
      submitButton.disabled = false;
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
