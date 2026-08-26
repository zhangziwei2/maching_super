const { createApp } = Vue;

createApp({
    data() {
        return {
            messages: [],
            userInput: '',
            isLoading: false,
            activeNav: 'newChat',
            abortController: null,
            sessionId: 'session_' + Date.now(),
            sessions: [],
            showHistorySidebar: false,
            isComposing: false,
            documents: [],
            documentsLoading: false,
            selectedFile: null,
            isUploading: false,
            uploadProgress: '',
            uploadSteps: [],
            uploadProgressCollapsed: false,
            activeUploadJobId: '',
            uploadPollTimer: null,
            deleteJobs: {},
            deletePollTimers: {},
            deleteRemoveTimers: {},
            token: localStorage.getItem('accessToken') || '',
            currentUser: null,
            authMode: 'login',
            authForm: {
                username: '',
                password: '',
                role: 'user',
                admin_code: ''
            },
            authLoading: false,
            showUploadMenu: false,
            baselineStatus: null,
            baselineConditions: [],
            showConditionModal: false,
            pendingBaselineFile: null,
            selectedCondition: '',
            isDiagnosing: false,
            // 知识图谱面板
            kgStats: null,
            kgQuery: '',
            kgResult: null,
            kgLoading: false,
            kgImportFile: null,
            kgImporting: false,
            kgImportResult: null,
            kgPanelTab: 'query',
        };
    },
    computed: {
        isAuthenticated() {
            return !!this.token && !!this.currentUser;
        },
        isAdmin() {
            return this.currentUser?.role === 'admin';
        },
    },
    async mounted() {
        this.configureMarked();
        if (this.token) {
            try {
                await this.fetchMe();
            } catch (_) {
                this.handleLogout();
            }
        }

        // 进入页面时刷新基线状态显示
        this.refreshBaselineStatus();
        

        // 点击弹出菜单外部时关闭
        document.addEventListener('mousedown', (e) => {
            const target = e.target;
            if (!target.closest('.toolbar-btn-wrapper')) {
                this.showUploadMenu = false;
            }
        });
    },
    beforeUnmount() {
        this.stopUploadJobPolling();
        this.stopAllDeleteJobPolling();
        Object.values(this.deleteRemoveTimers).forEach(timer => clearTimeout(timer));
    },
    methods: {
        configureMarked() {
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: 'hljs language-',
                breaks: true,
                gfm: true
            });
        },

        parseMarkdown(text) {
            return marked.parse(text);
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        authHeaders(extra = {}) {
            const headers = { ...extra };
            if (this.token) {
                headers.Authorization = `Bearer ${this.token}`;
            }
            return headers;
        },

        async authFetch(url, options = {}) {
            const opts = { ...options };
            opts.headers = this.authHeaders(opts.headers || {});
            const response = await fetch(url, opts);
            if (response.status === 401) {
                this.handleLogout();
                throw new Error('登录已过期，请重新登录');
            }
            return response;
        },

        async fetchMe() {
            const response = await this.authFetch('/auth/me');
            if (!response.ok) {
                throw new Error('认证失败');
            }
            this.currentUser = await response.json();
        },

        async handleAuthSubmit() {
            if (this.authLoading) return;
            const username = this.authForm.username.trim();
            const password = this.authForm.password.trim();
            if (!username || !password) {
                alert('用户名和密码不能为空');
                return;
            }

            this.authLoading = true;
            try {
                const endpoint = this.authMode === 'login' ? '/auth/login' : '/auth/register';
                const payload = {
                    username,
                    password
                };
                if (this.authMode === 'register') {
                    payload.role = this.authForm.role;
                    payload.admin_code = this.authForm.admin_code || null;
                }

                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || '认证失败');
                }

                this.token = data.access_token;
                this.currentUser = { username: data.username, role: data.role };
                localStorage.setItem('accessToken', this.token);
                this.authForm.password = '';
                this.authForm.admin_code = '';
                this.messages = [];
                this.sessionId = 'session_' + Date.now();
                this.activeNav = 'newChat';
            } catch (error) {
                alert(error.message);
            } finally {
                this.authLoading = false;
            }
        },

        handleLogout() {
            this.token = '';
            this.currentUser = null;
            this.messages = [];
            this.sessions = [];
            this.documents = [];
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
            localStorage.removeItem('accessToken');
        },

        // ========== 技能 ==========
        toggleUploadMenu() {
            this.showUploadMenu = !this.showUploadMenu;
        },

        triggerUpload(type) {
            this.showUploadMenu = false;
            if (type === 'image') this.$refs.imageInput.click();
            else if (type === 'csv_baseline') this.$refs.csvBaselineInput.click();
            else if (type === 'pdf') this.$refs.pdfInput.click();
            else if (type === 'chatter') this.$refs.chatterCsvInput.click();
            else if (type === 'monitor') this.$refs.monitorCsvInput.click();
        },

        async onUploadImage(event) {
            const file = event.target.files?.[0];
            if (!file) return;
            const form = new FormData();
            form.append('file', file);
            try {
                const resp = await this.authFetch('/upload/image', { method: 'POST', body: form });
                if (resp.ok) {
                    const data = await resp.json();
                    this.messages.push({ text: `🖼️ [图片] ${data.filename} — 上传成功`, isUser: true });
                } else {
                    const err = await resp.json().catch(() => ({}));
                    alert('上传失败：' + (err.detail || '未知错误'));
                }
            } catch (e) {
                alert('上传失败：' + e.message);
            }
            event.target.value = '';
        },

        async _runDiagnosis(file, mode, label) {
            const endpoint = mode === 'chatter_monitor' ? '/diagnose/monitor' : `/diagnose/chatter?mode=${mode}`;
            const form = new FormData();
            form.append('file', file);
            this.isDiagnosing = true;
            try {
                const resp = await this.authFetch(endpoint, { method: 'POST', body: form });
                if (resp.ok) {
                    const data = await resp.json();
                    let report = data.report || '诊断完成';
                    if (typeof report === 'object') report = JSON.stringify(report, null, 2);
                    this.messages.push({
                        text: `📊 [CSV] ${data.filename}\n━━━ ${label} ━━━\n\n${report}`,
                        isUser: false
                    });
                } else {
                    const err = await resp.json().catch(() => ({}));
                    alert('诊断失败：' + (err.detail || '未知错误'));
                }
            } catch (e) {
                alert('诊断失败：' + e.message);
            } finally {
                this.isDiagnosing = false;
            }
        },

        async onChatterDiagnosis(event) {
            const file = event.target.files?.[0];
            if (!file) return;
            await this._runDiagnosis(file, 'chatter_comprehensive', '🔬 颤振综合诊断');
            event.target.value = '';
        },

        async onMonitorDiagnosis(event) {
            const file = event.target.files?.[0];
            event.target.value = '';
            if (!file) return;
            await this._runDiagnosis(file, 'chatter_monitor', '🚦 实时监测');
        },

        async onRegisterBaseline(event) {
            const file = event.target.files?.[0];
            event.target.value = '';
            if (!file) return;
            // 先读取 CSV 第6列工况，列出可选标签供下拉选择
            const probe = new FormData();
            probe.append('file', file);
            try {
                const resp = await this.authFetch('/baseline/conditions', { method: 'POST', body: probe });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    alert('读取工况失败：' + (err.detail || '未知错误'));
                    return;
                }
                const data = await resp.json();
                const conds = data.conditions || [];
                if (!conds.length) {
                    // 无工况列：直接全段注册，并提示基线污染风险
                    await this._registerBaseline(file, '');
                    this.messages.push({
                        text: '⚠️ 该 CSV 未检测到第6列工况标签，基线将使用「全部段」。\n如含空载/颤振，基线会被污染，建议上传带工况列的 CSV。',
                        isUser: false
                    });
                    return;
                }
                // 多工况：弹出下拉选择（必须手选）
                this.baselineConditions = conds;
                this.pendingBaselineFile = file;
                this.selectedCondition = '';
                this.showConditionModal = true;
            } catch (e) {
                alert('读取工况失败：' + e.message);
            }
        },

        async _registerBaseline(file, condition) {
            const form = new FormData();
            form.append('file', file);
            const qs = [];
            if (condition) qs.push('condition=' + encodeURIComponent(condition));
            const url = '/baseline/register' + (qs.length ? '?' + qs.join('&') : '');
            try {
                const resp = await this.authFetch(url, { method: 'POST', body: form });
                if (resp.ok) {
                    const data = await resp.json();
                    let msg = `📋 [基线注册] ${file.name}`;
                    if (data.status === 'ok') {
                        msg += `\n✅ ${data.message}`;
                        msg += `\n📊 使用 ${data.used_segments}/${data.total_segments} 段`;
                        if (data.meta?.filter) msg += `\n🎯 筛选: ${data.meta.filter}`;
                        if (data.calibration) {
                            const c = data.calibration;
                            msg += `\n📐 自适应阈值: 关注≥${c.score_alert.toFixed(2)} / 报警≥${c.score_alarm.toFixed(2)}`;
                            msg += `\n   （依据基线自身波动标定，避免健康段被误报）`;
                        }
                    } else {
                        msg += `\n⚠️ ${data.message || '注册异常'}`;
                    }
                    this.messages.push({ text: msg, isUser: false });
                    await this.refreshBaselineStatus();
                } else {
                    const err = await resp.json().catch(() => ({}));
                    alert('基线注册失败：' + (err.detail || '未知错误'));
                }
            } catch (e) {
                alert('基线注册失败：' + e.message);
            }
        },

        async confirmRegisterBaseline() {
            const file = this.pendingBaselineFile;
            const raw = this.selectedCondition;
            this.showConditionModal = false;
            this.selectedCondition = '';
            this.pendingBaselineFile = null;
            if (!file || !raw) return;  // 未选择则取消
            const cond = raw === '__ALL__' ? '' : raw;
            await this._registerBaseline(file, cond);
        },

        async onClearBaseline() {
            if (!confirm('确定清除设备基线？清除后需重新注册基线才能实时监控。')) return;
            try {
                const resp = await this.authFetch('/baseline/reset', { method: 'POST' });
                if (resp.ok) {
                    const data = await resp.json();
                    this.messages.push({ text: `🧹 [基线清零] ${data.message || '已完成'}`, isUser: false });
                    await this.refreshBaselineStatus();
                } else {
                    const err = await resp.json().catch(() => ({}));
                    alert('基线清除失败：' + (err.detail || '未知错误'));
                }
            } catch (e) {
                alert('基线清除失败：' + e.message);
            }
        },

        async refreshBaselineStatus() {
            try {
                const resp = await this.authFetch('/baseline/info');
                if (resp.ok) {
                    this.baselineStatus = await resp.json();
                }
            } catch (_) { /* 忽略状态查询失败 */ }
        },

        async onUploadPDF(event) {
            const file = event.target.files?.[0];
            if (!file) return;
            const form = new FormData();
            form.append('file', file);
            try {
                const resp = await this.authFetch('/documents/upload/async', { method: 'POST', body: form });
                if (resp.ok) {
                    const data = await resp.json();
                    this.messages.push({
                        text: `📄 [PDF] ${data.filename} — 已提交后台解析入库。任务ID: ${data.job_id}`,
                        isUser: true
                    });
                    if (data.job_id) {
                        this.activeUploadJobId = data.job_id;
                        this.selectedFile = { name: data.filename };
                        this.uploadSteps = this.createUploadSteps();
                        this.uploadProgressCollapsed = false;
                        this.startUploadJobPolling(data.job_id);
                    }
                } else {
                    const err = await resp.json().catch(() => ({}));
                    alert('PDF 上传失败：' + (err.detail || '未知错误'));
                }
            } catch (e) {
                alert('PDF 上传失败：' + e.message);
            }
            event.target.value = '';
        },

        handleCompositionStart() {
            this.isComposing = true;
        },

        handleCompositionEnd() {
            this.isComposing = false;
        },

        handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },

        handleStop() {
            if (this.abortController) {
                this.abortController.abort();
            }
        },

        async handleSend() {
            if (!this.isAuthenticated) {
                alert('请先登录');
                return;
            }

            const text = this.userInput.trim();
            if (!text || this.isLoading || this.isComposing) return;

            this.messages.push({
                text: text,
                isUser: true
            });

            this.userInput = '';
            this.$nextTick(() => {
                this.resetTextareaHeight();
                this.scrollToBottom();
            });

            this.isLoading = true;
            this.messages.push({
                text: '',
                isUser: false,
                isThinking: true,
                ragTrace: null,
                ragSteps: []
            });
            const botMsgIdx = this.messages.length - 1;

            this.abortController = new AbortController();

            try {
                const response = await this.authFetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        session_id: this.sessionId,
                    }),
                    signal: this.abortController.signal,
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);

                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'content') {
                                    if (this.messages[botMsgIdx].isThinking) {
                                        this.messages[botMsgIdx].isThinking = false;
                                    }
                                    this.messages[botMsgIdx].text += data.content;
                                } else if (data.type === 'trace') {
                                    this.messages[botMsgIdx].ragTrace = data.rag_trace;
                                } else if (data.type === 'rag_step') {
                                    if (!this.messages[botMsgIdx].ragSteps) {
                                        this.messages[botMsgIdx].ragSteps = [];
                                    }
                                    this.messages[botMsgIdx].ragSteps.push(data.step);
                                } else if (data.type === 'error') {
                                    this.messages[botMsgIdx].isThinking = false;
                                    this.messages[botMsgIdx].text += `\n\n${data.content}`;
                                }
                            } catch (e) {
                                console.warn('SSE parse error:', e);
                            }
                        }
                    }
                    this.$nextTick(() => this.scrollToBottom());
                }

            } catch (error) {
                if (error.name === 'AbortError') {
                    this.messages[botMsgIdx].isThinking = false;
                    if (!this.messages[botMsgIdx].text) {
                        this.messages[botMsgIdx].text = '(已终止回答)';
                    } else {
                        this.messages[botMsgIdx].text += '\n\n_(回答已被终止)_';
                    }
                } else {
                    this.messages[botMsgIdx].isThinking = false;
                    this.messages[botMsgIdx].text = `喵呜... 出了点问题：${error.message}`;
                }
            } finally {
                this.isLoading = false;
                this.abortController = null;
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        autoResize(event) {
            const textarea = event.target;
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        },

        resetTextareaHeight() {
            if (this.$refs.textarea) {
                this.$refs.textarea.style.height = 'auto';
            }
        },

        scrollToBottom() {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
            }
        },

        handleNewChat() {
            if (!this.isAuthenticated) return;
            this.messages = [];
            this.sessionId = 'session_' + Date.now();
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
        },

        handleClearChat() {
            if (confirm('确定要清空当前对话吗？喵？')) {
                this.messages = [];
            }
        },

        async handleHistory() {
            if (!this.isAuthenticated) return;
            this.activeNav = 'history';
            this.showHistorySidebar = true;
            try {
                const response = await this.authFetch('/sessions');
                if (!response.ok) {
                    throw new Error('Failed to load sessions');
                }
                const data = await response.json();
                this.sessions = data.sessions;
            } catch (error) {
                alert('加载历史记录失败：' + error.message);
            }
        },

        async loadSession(sessionId) {
            this.sessionId = sessionId;
            this.showHistorySidebar = false;
            this.activeNav = 'newChat';

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`);
                if (!response.ok) {
                    throw new Error('Failed to load session messages');
                }
                const data = await response.json();
                this.messages = data.messages.map(msg => ({
                    text: msg.content,
                    isUser: msg.type === 'human',
                    ragTrace: msg.rag_trace || null
                }));

                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            } catch (error) {
                alert('加载会话失败：' + error.message);
                this.messages = [];
            }
        },

        async deleteSession(sessionId) {
            if (!confirm(`确定要删除会话 "${sessionId}" 吗？`)) {
                return;
            }

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`, {
                    method: 'DELETE'
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || 'Delete failed');
                }

                this.sessions = this.sessions.filter(s => s.session_id !== sessionId);

                if (this.sessionId === sessionId) {
                    this.messages = [];
                    this.sessionId = 'session_' + Date.now();
                    this.activeNav = 'newChat';
                }

                if (payload.message) {
                    alert(payload.message);
                }
            } catch (error) {
                alert('删除会话失败：' + error.message);
            }
        },

        handleSettings() {
            if (!this.isAdmin) {
                alert('仅管理员可访问文档管理');
                return;
            }
            this.activeNav = 'settings';
            this.showHistorySidebar = false;
            this.loadDocuments();
        },

        // ========== 知识图谱 ==========
        async openKgPanel(tab = 'query') {
            if (!this.isAuthenticated) return;
            this.activeNav = 'kg';
            this.showHistorySidebar = false;
            this.showUploadMenu = false;
            this.kgPanelTab = tab;
            if (!this.kgStats) {
                this.loadKgStats();
            }
        },

        async loadKgStats() {
            try {
                const resp = await this.authFetch('/kg/stats');
                if (resp.ok) {
                    this.kgStats = await resp.json();
                }
            } catch (e) {
                // 图谱服务不可用时静默忽略
            }
        },

        async runKgQuery() {
            const q = (this.kgQuery || '').trim();
            if (!q || this.kgLoading) return;
            this.kgLoading = true;
            try {
                const resp = await this.authFetch('/kg/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: q, hops: 2, top_k: 10 })
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.detail || '查询失败');
                }
                this.kgResult = await resp.json();
            } catch (e) {
                alert('图谱查询失败：' + e.message);
            } finally {
                this.kgLoading = false;
            }
        },

        onKgTriplesSelect(event) {
            this.kgImportFile = event.target.files?.[0] || null;
            this.kgImportResult = null;
        },

        async uploadKgTriples() {
            if (!this.kgImportFile || this.kgImporting) return;
            this.kgImporting = true;
            try {
                const form = new FormData();
                form.append('file', this.kgImportFile);
                form.append('llm_validate', 'true');
                const resp = await this.authFetch('/kg/triples/import', { method: 'POST', body: form });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.detail || '导入失败');
                }
                this.kgImportResult = await resp.json();
                this.loadKgStats();
                alert(`导入完成：成功 ${this.kgImportResult.imported} 条，拒绝 ${this.kgImportResult.rejected} 条`);
            } catch (e) {
                alert('三元组导入失败：' + e.message);
            } finally {
                this.kgImporting = false;
            }
        },

        mergeDocumentsWithActiveDeletes(nextDocuments) {
            const merged = Array.isArray(nextDocuments) ? [...nextDocuments] : [];
            Object.keys(this.deleteJobs).forEach(filename => {
                const job = this.deleteJobs[filename];
                if (!job || job.status === 'failed') return;
                const exists = merged.some(doc => doc.filename === filename);
                if (!exists) {
                    const currentDoc = this.documents.find(doc => doc.filename === filename);
                    if (currentDoc) {
                        merged.push(currentDoc);
                    }
                }
            });
            return merged;
        },

        async loadDocuments() {
            this.documentsLoading = true;
            try {
                const response = await this.authFetch('/documents');
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || 'Failed to load documents');
                }
                const data = await response.json();
                this.documents = this.mergeDocumentsWithActiveDeletes(data.documents);
            } catch (error) {
                alert('加载文档列表失败：' + error.message);
            } finally {
                this.documentsLoading = false;
            }
        },

        handleFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.selectedFile = files[0];
                this.uploadProgress = '';
                this.uploadSteps = this.createUploadSteps();
                this.uploadProgressCollapsed = false;
                this.activeUploadJobId = '';
            }
        },

        createUploadSteps() {
            return [
                { key: 'upload', label: '文档上传', percent: 0, status: 'pending', message: '' },
                { key: 'vector_store', label: '向量化入库', percent: 0, status: 'pending', message: '' },
            ];
        },

        updateUploadStep(key, percent, status = 'running', message = '') {
            if (!this.uploadSteps.length) {
                this.uploadSteps = this.createUploadSteps();
            }
            const idx = this.uploadSteps.findIndex(step => step.key === key);
            if (idx === -1) return;
            this.uploadSteps[idx] = {
                ...this.uploadSteps[idx],
                percent: Math.max(0, Math.min(100, Math.round(percent || 0))),
                status,
                message
            };
        },

        uploadFileWithProgress(file) {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                const formData = new FormData();
                formData.append('file', file);

                xhr.open('POST', '/documents/upload/async');
                const headers = this.authHeaders();
                Object.entries(headers).forEach(([key, value]) => xhr.setRequestHeader(key, value));

                xhr.upload.onprogress = (event) => {
                    if (!event.lengthComputable) return;
                    const percent = Math.round((event.loaded / event.total) * 100);
                    this.updateUploadStep('upload', percent, 'running', `已上传 ${percent}%`);
                };

                xhr.onload = () => {
                    if (xhr.status === 401) {
                        this.handleLogout();
                        reject(new Error('登录已过期，请重新登录'));
                        return;
                    }

                    let data = {};
                    try {
                        data = JSON.parse(xhr.responseText || '{}');
                    } catch (e) {
                        reject(new Error('上传响应解析失败'));
                        return;
                    }

                    if (xhr.status < 200 || xhr.status >= 300) {
                        reject(new Error(data.detail || `HTTP ${xhr.status}`));
                        return;
                    }

                    this.updateUploadStep('upload', 100, 'completed', '文档上传完成');
                    resolve(data);
                };

                xhr.onerror = () => reject(new Error('上传请求失败'));
                xhr.onabort = () => reject(new Error('上传已取消'));
                xhr.send(formData);
            });
        },

        syncUploadJob(job) {
            this.activeUploadJobId = job.job_id;
            this.uploadProgress = job.message || '';
            if (Array.isArray(job.steps)) {
                this.uploadSteps = job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                }));
            }
            // 入库成功后自动收起步骤明细，保留摘要供用户再次展开查看。
            if (job.status === 'completed') {
                this.uploadProgressCollapsed = true;
            }
        },

        toggleUploadProgressCollapsed() {
            this.uploadProgressCollapsed = !this.uploadProgressCollapsed;
        },

        // 将后端原始报错转成用户能看懂的中文提示（尤其是 Milvus 不可达这类常见故障）
        friendlyUploadError(err) {
            if (!err) return '文档处理失败';
            const s = String(err);
            if (/milvus/i.test(s) && /(19530|connect|unavailable)/i.test(s)) {
                return 'Milvus 向量库不可用，请先启动 Milvus（docker compose up -d standalone）后再重试';
            }
            return s;
        },

        stopUploadJobPolling() {
            if (this.uploadPollTimer) {
                clearInterval(this.uploadPollTimer);
                this.uploadPollTimer = null;
            }
        },

        startUploadJobPolling(jobId) {
            this.stopUploadJobPolling();

            const poll = async () => {
                try {
                    const response = await this.authFetch(`/documents/upload/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to load upload job');
                    }

                    const job = await response.json();
                    this.syncUploadJob(job);

                    if (job.status === 'completed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                        this.selectedFile = null;
                        if (this.$refs.fileInput) {
                            this.$refs.fileInput.value = '';
                        }
                        await this.loadDocuments();
                    } else if (job.status === 'failed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                        // 明确提示失败原因并复位上传状态，避免界面停留在失败步骤看起来像"卡住"
                        const errMsg = this.friendlyUploadError(job.error || job.message || '文档处理失败');
                        this.uploadProgress = '文档处理失败：' + errMsg;
                        alert('文档处理失败：' + errMsg);
                        this.selectedFile = null;
                        this.activeUploadJobId = '';
                        if (this.$refs.fileInput) {
                            this.$refs.fileInput.value = '';
                        }
                    }
                } catch (error) {
                    this.uploadProgress = '进度查询失败：' + error.message;
                    this.stopUploadJobPolling();
                    this.isUploading = false;
                }
            };

            poll();
            this.uploadPollTimer = setInterval(poll, 1000);
        },

        async uploadDocument() {
            if (!this.selectedFile) {
                alert('请先选择文件');
                return;
            }

            this.isUploading = true;
            this.uploadProgress = '正在上传...';
            this.uploadSteps = this.createUploadSteps();
            this.uploadProgressCollapsed = false;
            this.updateUploadStep('upload', 0, 'running', '准备上传');

            try {
                const data = await this.uploadFileWithProgress(this.selectedFile);
                this.uploadProgress = data.message;
                this.activeUploadJobId = data.job_id;
                this.startUploadJobPolling(data.job_id);
            } catch (error) {
                this.updateUploadStep('upload', 100, 'failed', error.message);
                this.uploadProgress = '上传失败：' + error.message;
                this.isUploading = false;
            }
        },

        createDeleteSteps() {
            return [
                { key: 'prepare', label: '准备删除', percent: 0, status: 'pending', message: '' },
                { key: 'bm25', label: '同步 BM25 统计', percent: 0, status: 'pending', message: '' },
                { key: 'milvus', label: '删除向量数据', percent: 0, status: 'pending', message: '' },
                { key: 'parent_store', label: '删除父级分块', percent: 0, status: 'pending', message: '' },
            ];
        },

        isDeletingDocument(filename) {
            const job = this.deleteJobs[filename];
            return job && job.status === 'running';
        },

        isDeleteActionLocked(filename) {
            const job = this.deleteJobs[filename];
            return job && (job.status === 'running' || job.status === 'completed');
        },

        getDeleteButtonIcon(filename) {
            const job = this.deleteJobs[filename];
            if (job?.status === 'running') return 'fas fa-spinner fa-spin';
            if (job?.status === 'completed') return 'fas fa-check';
            return 'fas fa-trash';
        },

        setDeleteJob(filename, nextJob) {
            this.deleteJobs = {
                ...this.deleteJobs,
                [filename]: {
                    ...(this.deleteJobs[filename] || {}),
                    ...nextJob
                }
            };
        },

        syncDeleteJob(filename, job) {
            const current = this.deleteJobs[filename] || {};
            // 后端返回统一的步骤结构，前端只负责同步到当前文档行内卡片。
            this.setDeleteJob(filename, {
                jobId: job.job_id,
                status: job.status,
                message: job.message || '',
                collapsed: job.status === 'completed' ? true : Boolean(current.collapsed),
                steps: Array.isArray(job.steps) ? job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                })) : this.createDeleteSteps()
            });
        },

        toggleDeleteJobCollapsed(filename) {
            const job = this.deleteJobs[filename];
            if (!job) return;
            this.setDeleteJob(filename, { collapsed: !job.collapsed });
        },

        stopDeleteJobPolling(filename) {
            const timer = this.deletePollTimers[filename];
            if (!timer) return;
            clearInterval(timer);
            const { [filename]: _removed, ...rest } = this.deletePollTimers;
            this.deletePollTimers = rest;
        },

        stopAllDeleteJobPolling() {
            Object.keys(this.deletePollTimers).forEach(filename => this.stopDeleteJobPolling(filename));
        },

        clearDeleteRemovalTimer(filename) {
            const timer = this.deleteRemoveTimers[filename];
            if (!timer) return;
            clearTimeout(timer);
            const { [filename]: _removed, ...rest } = this.deleteRemoveTimers;
            this.deleteRemoveTimers = rest;
        },

        scheduleDeletedDocumentRemoval(filename) {
            this.clearDeleteRemovalTimer(filename);
            // 删除完成后先保留 3 秒摘要，再从当前列表移除并刷新后端状态。
            const timer = setTimeout(async () => {
                this.documents = this.documents.filter(doc => doc.filename !== filename);
                const { [filename]: _job, ...jobs } = this.deleteJobs;
                const { [filename]: _timer, ...timers } = this.deleteRemoveTimers;
                this.deleteJobs = jobs;
                this.deleteRemoveTimers = timers;
                await this.loadDocuments();
            }, 3000);
            this.deleteRemoveTimers = {
                ...this.deleteRemoveTimers,
                [filename]: timer
            };
        },

        startDeleteJobPolling(filename, jobId) {
            this.stopDeleteJobPolling(filename);

            const poll = async () => {
                try {
                    const response = await this.authFetch(`/documents/delete/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to load delete job');
                    }

                    const job = await response.json();
                    this.syncDeleteJob(filename, job);

                    if (job.status === 'completed') {
                        this.stopDeleteJobPolling(filename);
                        this.scheduleDeletedDocumentRemoval(filename);
                    } else if (job.status === 'failed') {
                        this.stopDeleteJobPolling(filename);
                    }
                } catch (error) {
                    this.setDeleteJob(filename, {
                        status: 'failed',
                        message: '删除进度查询失败：' + error.message,
                        collapsed: false,
                        steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps()
                    });
                    this.stopDeleteJobPolling(filename);
                }
            };

            poll();
            this.deletePollTimers = {
                ...this.deletePollTimers,
                [filename]: setInterval(poll, 1000)
            };
        },

        async deleteDocument(filename) {
            if (this.isDeletingDocument(filename)) {
                return;
            }
            if (!confirm(`确定要删除文档 "${filename}" 吗？这将同时删除 Milvus 中的所有相关向量。`)) {
                return;
            }

            this.clearDeleteRemovalTimer(filename);
            this.setDeleteJob(filename, {
                status: 'running',
                message: '正在提交删除任务...',
                collapsed: false,
                steps: this.createDeleteSteps().map(step => (
                    step.key === 'prepare'
                        ? { ...step, percent: 1, status: 'running', message: '正在提交删除任务' }
                        : step
                ))
            });

            try {
                const response = await this.authFetch(`/documents/delete/async/${encodeURIComponent(filename)}`, {
                    method: 'DELETE'
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || 'Delete failed');
                }

                const data = await response.json();
                this.setDeleteJob(filename, {
                    jobId: data.job_id,
                    status: 'running',
                    message: data.message || `正在删除 ${filename}`,
                    collapsed: false
                });
                this.startDeleteJobPolling(filename, data.job_id);

            } catch (error) {
                this.setDeleteJob(filename, {
                    status: 'failed',
                    message: '删除文档失败：' + error.message,
                    collapsed: false,
                    steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps()
                });
            }
        },

        getFileIcon(fileType) {
            if (fileType === 'PDF') {
                return 'fas fa-file-pdf';
            } else if (fileType === 'Word') {
                return 'fas fa-file-word';
            } else if (fileType === 'Excel') {
                return 'fas fa-file-excel';
            }
            return 'fas fa-file';
        }
    },
    watch: {
        messages: {
            handler() {
                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            },
            deep: true
        }
    }
}).mount('#app');
