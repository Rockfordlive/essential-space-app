import React, { useState, useEffect, useCallback, useRef, useImperativeHandle, forwardRef } from 'react';

// --- Helper Functions ---
// Formats an ISO date string into a more readable "dd/mm/yy, HH:MM" format.
const formatDate = (isoString) => {
    if (!isoString) return '';
    const date = new Date(isoString);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = String(date.getFullYear()).slice(-2);
    const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return `${day}/${month}/${year}, ${time}`;
};

// Formats an ISO date string into "Month Day" (e.g., "Aug 23").
const formatDueDate = (isoDateString) => {
    if (!isoDateString) return '';
    const date = new Date(isoDateString);
    // Add time to counteract timezone shifts that could change the date
    const adjustedDate = new Date(date.getTime() + date.getTimezoneOffset() * 60000);
    return adjustedDate.toLocaleDateString([], { month: 'short', day: 'numeric' });
};


// Converts a Date object to a "YYYY-MM-DD" string, useful for date inputs and comparisons.
const toISODateString = (date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
};

// --- Offline Content Processing ---
/**
 * Creates a simple extractive summary by finding the most "important" sentence.
 * This works entirely offline.
 * @param {string} noteText - The text to be summarized.
 * @returns {object} - A structured object with title and summary.
 */
const createInAppSummary = (noteText) => {
    if (!noteText || noteText.trim() === '') {
        return { title: "New Note", summary: "No content to summarize.", extractedText: noteText, highlights: [], tags: [] };
    }

    const stopWords = new Set(['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']);
    const sentences = noteText.match(/[^.!?]+[.!?]+/g) || [noteText];
    const wordFrequencies = {};
    noteText.toLowerCase().replace(/\[|\]/g, ' ').split(/\s+/).forEach(word => {
        const cleanWord = word.replace(/[.,!?]/g, '');
        if (!stopWords.has(cleanWord) && cleanWord.length > 0) {
            wordFrequencies[cleanWord] = (wordFrequencies[cleanWord] || 0) + 1;
        }
    });

    let maxScore = 0;
    let bestSentence = sentences[0];
    sentences.forEach(sentence => {
        let score = 0;
        sentence.toLowerCase().split(/\s+/).forEach(word => {
            const cleanWord = word.replace(/[.,!?]/g, '');
            if (wordFrequencies[cleanWord]) {
                score += wordFrequencies[cleanWord];
            }
        });
        if (score > maxScore) {
            maxScore = score;
            bestSentence = sentence;
        }
    });

    const title = noteText.split('\n')[0].substring(0, 40) || "New Note";
    return {
        title,
        summary: bestSentence.trim(),
        extractedText: noteText,
        highlights: [],
        tags: []
    };
};


// --- API Integration ---
const callGeminiApi = async (payload, retries = 3, delay = 1000) => {
    const model = 'gemini-2.5-flash-preview-05-20';
    const apiKey = ""; // This will be handled by the environment
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                const errorBody = await response.text();
                console.error("API Error Response:", errorBody);
                throw new Error(`API request failed with status ${response.status}`);
            }
            const result = await response.json();
            if (result.candidates?.[0]?.content?.parts?.[0]) {
                return result;
            } else {
                console.warn("Invalid response structure from Gemini API:", result);
                throw new Error("Invalid response structure from Gemini API");
            }
        } catch (error) {
            console.error(`Attempt ${i + 1} failed:`, error.message);
            if (i === retries - 1) { throw error; }
            await new Promise(res => setTimeout(res, delay * Math.pow(2, i)));
        }
    }
    throw new Error("API request failed after all retries.");
};

/**
 * Main content processing controller. Tries Gemini API first, then falls back to offline summary.
 * @param {object} details - An object containing either a 'note' string or 'imageData' string.
 * @returns {Promise<object>} - A structured object with title, summary, highlights, and tasks.
 */
const processContent = async (details) => {
    if (details.imageData) {
        try {
            const prompt = `You are an intelligent assistant that analyzes images. Extract information from the provided screenshot.
1.  **Analyze the content**: Understand what the image is about.
2.  **Extract all text**: Perform OCR to get all visible text.
3.  **Generate a title**: Create a short, descriptive title (max 5 words).
4.  **Summarize**: Write a concise two-sentence summary.
5.  **Identify Tags**: Find 2-3 key pieces of information or keywords to act as tags.
6.  **Suggest Highlights**: Based on the content, suggest concrete action items or highlights. If a date/time is associated, include it.

Respond ONLY with a valid JSON object following this exact schema. Do not include any other text or markdown formatting.
The current date is ${new Date().toDateString()}.`;
            const base64ImageData = details.imageData.split(',')[1];
            const payload = {
                contents: [{ parts: [{ text: prompt }, { inlineData: { mimeType: "image/png", data: base64ImageData } }] }],
                generationConfig: {
                    responseMimeType: "application/json",
                    responseSchema: {
                        type: "OBJECT",
                        properties: {
                            "title": { "type": "STRING" },
                            "summary": { "type": "STRING" },
                            "extractedText": { "type": "STRING" },
                            "tags": { "type": "ARRAY", "items": { "type": "STRING" } },
                            "highlights": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "description": { "type": "STRING" },
                                        "date": { "type": "STRING", "description": "Date in YYYY-MM-DD format, if applicable" }
                                    },
                                    "required": ["description"]
                                }
                            }
                        },
                        required: ["title", "summary", "extractedText", "tags", "highlights"]
                    }
                }
            };
            const result = await callGeminiApi(payload);
            const data = JSON.parse(result.candidates[0].content.parts[0].text);
            return {
                title: data.title || "Screenshot",
                summary: data.summary || "No summary available.",
                extractedText: data.extractedText || "",
                highlights: data.highlights || [],
                tags: data.tags || [],
            };
        } catch (error) {
            console.error("Error processing image with Gemini, creating fallback.", error);
            return { title: "Screenshot", summary: "Could not analyze image content.", extractedText: "Analysis failed. No text could be extracted.", highlights: [], tags: [] };
        }
    }

    if (details.note) {
        try {
            const prompt = `You are an intelligent assistant. Analyze the content and provide a concise title, a one-sentence summary, and 2-3 relevant keyword tags. Respond ONLY with a valid JSON object in the format: {"title": "string", "summary": "string", "tags": ["tag1", "tag2"]}.`;
            const payload = {
                contents: [{ parts: [{ text: `${prompt}\n\nNote:\n${details.note}` }] }],
                generationConfig: {
                    responseMimeType: "application/json",
                    responseSchema: { type: "OBJECT", properties: { "title": { "type": "STRING" }, "summary": { "type": "STRING" }, "tags": { "type": "ARRAY", "items": { "type": "STRING" } } }, required: ["title", "summary", "tags"] }
                }
            };
            const result = await callGeminiApi(payload);
            const data = JSON.parse(result.candidates[0].content.parts[0].text);
            return { ...data, extractedText: details.note, highlights: [] };
        } catch (error) {
            console.warn("Gemini API failed, falling back to offline summary.", error.message);
            return createInAppSummary(details.note);
        }
    }
    
    return { title: "New Item", summary: "", extractedText: "", highlights: [], tags: [] };
};


// --- Components ---

const ThemeToggleButton = ({ theme, toggleTheme }) => (
    <div className="flex items-center justify-center my-4">
        <label htmlFor="theme-toggle" className="flex items-center cursor-pointer">
            <div className="relative">
                <input type="checkbox" id="theme-toggle" className="sr-only" checked={theme === 'light'} onChange={toggleTheme} />
                <div className="block bg-gray-600 w-14 h-8 rounded-full"></div>
                <div className={`dot absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition-transform ${theme === 'light' ? 'transform translate-x-6' : ''}`}></div>
            </div>
            <div className="ml-3 text-sm font-medium">
                {theme === 'light' ? 'Light' : 'Dark'} Mode
            </div>
        </label>
    </div>
);

const CollectionMenu = ({ collection, onClose, onRemove, onDelete, position }) => {
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const menuRef = useRef(null);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                onClose();
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [onClose]);

    if (!collection) return null;

    return (
        <div ref={menuRef} style={{ top: position.top, left: position.left }} className="absolute z-50 bg-modal rounded-md shadow-lg w-48">
            {!showDeleteConfirm ? (
                <div className="py-1">
                    <a href="#" onClick={(e) => { e.preventDefault(); onRemove(collection.id); onClose(); }} className="block px-4 py-2 text-sm text-primary-text hover:bg-hover">Remove Collection</a>
                    <a href="#" onClick={(e) => { e.preventDefault(); setShowDeleteConfirm(true); }} className="block px-4 py-2 text-sm text-red-500 hover:bg-hover">Delete Collection</a>
                </div>
            ) : (
                 <div className="p-4">
                    <p className="text-sm text-secondary-text mb-4">Delete collection and all memories?</p>
                    <div className="flex justify-end space-x-2">
                        <button onClick={() => setShowDeleteConfirm(false)} className="px-3 py-1 rounded-md bg-surface hover:bg-hover text-xs font-semibold">Cancel</button>
                        <button onClick={() => {onDelete(collection.id); onClose();}} className="px-3 py-1 rounded-md bg-red-600 hover:bg-red-700 text-white text-xs font-semibold">Delete</button>
                    </div>
                </div>
            )}
        </div>
    );
};


const Sidebar = ({ theme, toggleTheme, activePage, setActivePage, collections, onSaveCollection, onOpenCollectionMenu, onDropOnCollection, setSelectedCollectionId }) => {
    const [isCollectionOpen, setIsCollectionOpen] = useState(false);
    const [newCollectionName, setNewCollectionName] = useState('');
    const [showInput, setShowInput] = useState(false);
    const [dragOver, setDragOver] = useState(null);

    const handleCollectionClick = (e) => {
        e.preventDefault();
        setIsCollectionOpen(!isCollectionOpen);
        setActivePage('collection');
    };

    const handleSave = () => {
        if (newCollectionName.trim()) {
            onSaveCollection({
                id: Date.now(),
                name: newCollectionName,
                createdAt: new Date().toISOString(),
            });
            setNewCollectionName('');
            setShowInput(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleSave();
        } else if (e.key === 'Escape') {
            setNewCollectionName('');
            setShowInput(false);
        }
    };

    const handleDragOver = (e, id) => {
        e.preventDefault();
        setDragOver(id);
    };

    const handleDragLeave = () => {
        setDragOver(null);
    };

    const handleDrop = (e, id) => {
        const itemId = e.dataTransfer.getData('text/plain');
        onDropOnCollection(itemId, id);
        setDragOver(null);
    };
    
    return (
    <aside className="w-48 bg-sidebar p-4 flex flex-col space-y-4">
        <h1 className="text-xl font-bold tracking-tight px-2 mb-2">Essential Space</h1>
        <ThemeToggleButton theme={theme} toggleTheme={toggleTheme} />
        <nav className="flex flex-col space-y-2">
            <a href="#" onClick={() => { setActivePage('home'); setSelectedCollectionId(null); }} className={`nav-link ${activePage === 'home' ? 'active' : ''} flex items-center space-x-3 px-3 py-2 rounded-lg`}>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
                <span>Home</span>
            </a>
            <a href="#" onClick={() => setActivePage('todo')} className={`nav-link ${activePage === 'todo' ? 'active' : ''} flex items-center space-x-3 px-3 py-2 rounded-lg`}>
                 <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"></path></svg>
                <span>To-Do</span>
            </a>
            <a href="#" onClick={handleCollectionClick} className={`nav-link ${activePage === 'collection' ? 'active' : ''} flex items-center space-x-3 px-3 py-2 rounded-lg`}>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                <span>Collection</span>
            </a>
             {isCollectionOpen && (
                <div className="pl-4 mt-2 border-l-2 border-border-color">
                    {collections.map(collection => (
                        <div
                            key={collection.id}
                            className={`p-2 border-b border-border-color/50 flex justify-between items-center cursor-pointer collection-item ${dragOver === collection.id ? 'bg-accent' : ''}`}
                            onDragOver={(e) => handleDragOver(e, collection.id)}
                            onDragLeave={handleDragLeave}
                            onDrop={(e) => handleDrop(e, collection.id)}
                            onClick={() => setSelectedCollectionId(collection.id)}
                        >
                            <span className="truncate-2-words">{collection.name}</span>
                            <button onClick={(e) => { e.stopPropagation(); onOpenCollectionMenu(collection, e.currentTarget); }} className="p-1 rounded-full hover:bg-hover">
                                &hellip;
                            </button>
                        </div>
                    ))}
                     {showInput && (
                        <div className="flex items-center mt-2">
                            <input
                                type="text"
                                value={newCollectionName}
                                onChange={(e) => setNewCollectionName(e.target.value)}
                                onKeyDown={handleKeyDown}
                                className="flex-1 min-w-0 bg-input border border-border-color rounded-md p-1 text-sm focus:outline-none focus:ring-1 focus:ring-accent"
                                placeholder="New..."
                            />
                            <button onClick={handleSave} className="ml-2 p-1 bg-accent rounded-full text-white">
                                &#10003;
                            </button>
                        </div>
                    )}
                    <button onClick={() => setShowInput(true)} className="mt-2 w-full text-left text-sm text-secondary-text hover:text-primary-text">+ Add New</button>
                </div>
            )}
        </nav>
    </aside>
)};

const AddNote = ({ onSave, isProcessingScreenshot, onFullScreen }) => {
    const [note, setNote] = useState('');
    const [isProcessingNote, setIsProcessingNote] = useState(false);

    const handleSave = async () => {
        if (note.trim()) {
            setIsProcessingNote(true);
            const aiData = await processContent({ note });
            onSave({ note, type: 'memory' }, aiData);
            setNote('');
            setIsProcessingNote(false);
        }
    };

    const isProcessing = isProcessingNote || isProcessingScreenshot;

    return (
        <section>
            <h2 className="text-lg font-semibold mb-3">Add Note or Paste Screenshot</h2>
            <div className={`bg-surface rounded-lg p-4 space-y-3 transition-all min-h-[150px] flex flex-col items-center justify-center relative`}>
                
                {isProcessing ? (
                    <div className="text-center text-secondary-text">
                        <div className="processing-spinner inline-block"></div>
                        <p className="mt-2 text-sm font-semibold">✨ Analyzing Your Memory...</p>
                    </div>
                ) : (
                    <div className="w-full relative">
                        <textarea
                            value={note}
                            onChange={(e) => setNote(e.target.value)}
                            className="w-full bg-input border border-border-color rounded-md p-2 text-sm focus:outline-none focus:ring-1 focus:ring-accent"
                            rows="3"
                            placeholder="Enter your note or paste screenshot"
                        ></textarea>
                        <button onClick={() => onFullScreen(note)} className="absolute top-0 right-0 mt-2 mr-2 text-xs bg-transparent border border-white rounded px-2 py-1 text-white hover:bg-white hover:text-black">
                           Click here for full screen
                        </button>
                        <div className="flex items-center space-x-2 mt-2">
                            <button onClick={handleSave} className="w-full bg-accent hover:bg-blue-500 text-white font-semibold py-2 rounded-lg transition-colors border border-blue-400">Save Note</button>
                        </div>
                    </div>
                )}
            </div>
        </section>
    );
};

const FullScreenNoteModal = ({ initialNote, onSave, onClose }) => {
    const [note, setNote] = useState(initialNote);
    const [isProcessingNote, setIsProcessingNote] = useState(false);

    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose]);

    const handleSave = async () => {
        if (note.trim()) {
            setIsProcessingNote(true);
            const aiData = await processContent({ note });
            onSave({ note, type: 'memory' }, aiData);
            setNote('');
            setIsProcessingNote(false);
            onClose();
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(5px)' }}>
            <div className="bg-modal rounded-lg p-6 shadow-xl w-full max-w-2xl h-5/6 flex flex-col">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="font-semibold text-lg">Add Note</h3>
                    <button onClick={onClose} className="p-2 hover:bg-hover rounded-full">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>
                <textarea
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    className="flex-1 w-full bg-input border border-border-color rounded-md p-4 text-base focus:outline-none focus:ring-1 focus:ring-accent"
                    placeholder="Enter your note..."
                ></textarea>
                <div className="mt-4">
                    <button onClick={handleSave} disabled={isProcessingNote} className="w-full bg-accent hover:bg-blue-500 text-white font-semibold py-3 rounded-lg transition-colors border border-blue-400">
                        {isProcessingNote ? 'Saving...' : 'Save Note'}
                    </button>
                </div>
            </div>
        </div>
    );
};

const TodoList = ({ items, onToggle, onSelectTodo, onShowDatePicker }) => {
    const todos = items.filter(item => item.type === 'todo').sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    const buttonRefs = useRef({});

    if (todos.length === 0) {
        return <p className="text-secondary-text text-sm">No tasks here.</p>;
    }

    return (
        <div className="space-y-2 max-h-32 overflow-y-auto">
            {todos.map(item => (
                <div key={item.id} className="p-2 border-b border-border-color/50">
                    {item.note.split('\n').map((line, index) => {
                        const match = line.match(/-\s*\[( |x|X)\]\s*(.*)/);
                        if (!match) return null;
                        const text = match[2];
                        const isChecked = match[1].toLowerCase() === 'x';
                        const checkboxId = `center-todo-${item.id}-${index}`;
                        return (
                            <div key={index} className="todo-item flex items-center justify-between text-sm group" >
                                <div className="flex items-center cursor-pointer" onClick={() => onSelectTodo(item)}>
                                    <input
                                        type="checkbox"
                                        id={checkboxId}
                                        className="mr-2 accent-accent"
                                        checked={isChecked}
                                        onChange={(e) => {
                                            e.stopPropagation();
                                            onToggle(item.id, index);
                                        }}
                                    />
                                    <label htmlFor={checkboxId} className="flex items-center cursor-pointer">
                                        <span className={isChecked ? 'text-secondary-text line-through' : ''}>{text}</span>
                                    </label>
                                </div>
                                <div className="relative">
                                    {!item.deadline ? (
                                        <button ref={el => buttonRefs.current[item.id] = el} onClick={(e) => { e.stopPropagation(); onShowDatePicker(item.id, buttonRefs.current[item.id]); }} className="text-xs bg-input px-2 py-1 rounded-full text-white">Add Date</button>
                                    ) : (
                                        <span ref={el => buttonRefs.current[item.id] = el} onClick={(e) => { e.stopPropagation(); onShowDatePicker(item.id, buttonRefs.current[item.id]);}} className="text-xs bg-input px-2 py-1 rounded-full text-secondary-text cursor-pointer">{formatDueDate(item.deadline)}</span>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            ))}
        </div>
    );
};

const AddTodo = forwardRef(({ onSave, items, onToggle, onSelectTodo, onShowDatePicker }, ref) => {
    const [task, setTask] = useState('');
    const [selectedDate, setSelectedDate] = useState(null);
    const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);
    const datePickerRef = useRef(null);
    const taskInputRef = useRef(null);

    useImperativeHandle(ref, () => ({
        setFocusAndDate(date) {
            if (date) {
                setSelectedDate(date);
            }
            taskInputRef.current?.focus();
        }
    }));

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (datePickerRef.current && !datePickerRef.current.contains(event.target)) {
                setIsDatePickerOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleAdd = () => {
        if (task.trim()) {
            const aiData = { title: task, summary: '', highlights: [], tasks: [] };
            onSave({ note: `- [ ] ${task}`, type: 'todo', deadline: selectedDate }, aiData);
            setTask('');
            setSelectedDate(null);
        }
    };
    
    const handleKeyDown = (event) => {
        if (event.key === 'Enter') {
            handleAdd();
        }
    };

    return (
        <section>
            <h2 className="text-lg font-semibold mb-3">To-Do</h2>
            <div className="bg-surface rounded-lg p-4 space-y-2">
                <TodoList items={items} onToggle={onToggle} onSelectTodo={onSelectTodo} onShowDatePicker={onShowDatePicker} />
                <div className="flex items-center space-x-2 pt-2 border-t border-border-color relative">
                    <input
                        ref={taskInputRef}
                        type="text"
                        value={task}
                        onChange={(e) => setTask(e.target.value)}
                        onKeyDown={handleKeyDown}
                        className="flex-1 bg-input border border-border-color rounded-md p-2 text-sm focus:outline-none focus:ring-1 focus:ring-accent"
                        placeholder="Add a new task..."
                    />
                    <div className="relative" ref={datePickerRef}>
                        <button 
                            onClick={() => setIsDatePickerOpen(!isDatePickerOpen)} 
                            className={`bg-input border border-border-color rounded-md hover:bg-hover transition-all ${selectedDate ? 'ring-2 ring-accent px-3 py-2 text-xs' : 'p-2'}`}
                        >
                            {selectedDate ? (
                                <span className="font-semibold">{formatDueDate(selectedDate)}</span>
                            ) : (
                                <svg className="w-4 h-4 text-secondary-text" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd"></path></svg>
                            )}
                        </button>
                        {isDatePickerOpen && <DatePickerWidget onSelect={setSelectedDate} close={() => setIsDatePickerOpen(false)} />}
                    </div>
                    <button onClick={handleAdd} className="bg-accent hover:bg-blue-500 text-white font-semibold px-4 py-2 rounded-lg transition-colors">Add</button>
                </div>
            </div>
        </section>
    );
});

const DatePickerWidget = ({ onSelect, close, anchorRef }) => {
    const [date, setDate] = useState(new Date());
    const [position, setPosition] = useState({ top: 0, left: 0 });
    const pickerRef = useRef(null);

    useEffect(() => {
        // The anchorRef can be a ref object or a direct DOM element.
        const element = anchorRef?.current ? anchorRef.current : anchorRef;
        if (element) {
            const rect = element.getBoundingClientRect();
            setPosition({ top: rect.bottom + window.scrollY + 5, left: rect.left + window.scrollX });
        }
    }, [anchorRef]);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (pickerRef.current && !pickerRef.current.contains(event.target)) {
                close();
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [close]);


    const renderDays = () => {
        const year = date.getFullYear();
        const month = date.getMonth();
        const firstDay = new Date(year, month, 1).getDay();
        const daysInMonth = new Date(year, month + 1, 0).getDate();
        const days = [];

        for (let i = 0; i < firstDay; i++) {
            days.push(<div key={`empty-${i}`}></div>);
        }

        for (let i = 1; i <= daysInMonth; i++) {
            const dayDate = new Date(year, month, i);
            const dateStr = toISODateString(dayDate);
            days.push(
                <div
                    key={i}
                    onClick={() => { onSelect(dateStr); close(); }}
                    className="date-picker-day flex items-center justify-center text-xs rounded-full cursor-pointer hover:bg-hover"
                >
                    {i}
                </div>
            );
        }
        return days;
    };
    
    const style = anchorRef ? { position: 'fixed', top: `${position.top}px`, left: `${position.left}px` } : { position: 'absolute', bottom: '100%', right: 0 };

    return (
        <div ref={pickerRef} style={style} className="mb-2 w-64 rounded-lg p-2 shadow-lg z-50 bg-surface border border-border-color">
            <div className="flex justify-between items-center mb-2 px-1">
                <button onClick={() => setDate(new Date(date.setMonth(date.getMonth() - 1)))}>&lt;</button>
                <h3 className="font-semibold text-sm">{date.toLocaleString('default', { month: 'short' })} {date.getFullYear()}</h3>
                <button onClick={() => setDate(new Date(date.setMonth(date.getMonth() + 1)))}>&gt;</button>
            </div>
            <div className="grid grid-cols-7 text-center text-secondary-text text-xs mb-1">
                <div>Su</div><div>Mo</div><div>Tu</div><div>We</div><div>Th</div><div>Fr</div><div>Sa</div>
            </div>
            <div className="grid grid-cols-7 gap-1">{renderDays()}</div>
        </div>
    );
};

const MemoriesFeed = ({ items, onSelect, selectedCollectionId }) => {
    const [query, setQuery] = useState('');
    
    const memories = items.filter(item => {
        const inCollection = selectedCollectionId ? item.collectionId === selectedCollectionId : true;
        const matchesQuery = !query || item.title.toLowerCase().includes(query.toLowerCase()) || (item.tags && item.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase())));
        return item.type === 'memory' && inCollection && matchesQuery;
    }).sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    const handleDragStart = (e, item) => {
        e.dataTransfer.setData('text/plain', String(item.id));
        e.dataTransfer.effectAllowed = 'move';
    };

    return (
        <section className="flex-1 flex flex-col">
            <div className="flex justify-between items-center mb-3">
                <h2 className="text-lg font-semibold">Memories</h2>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="w-1/2 bg-surface border border-border-color rounded-full px-4 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-accent"
                    placeholder="Search memories..."
                />
            </div>
            {memories.length === 0 ? (
                <div className="text-center py-10">
                    <p className="text-lg font-medium text-secondary-text">Your space is empty.</p>
                </div>
            ) : (
                <div id="feed-container" className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1 overflow-y-auto pb-6">
                    {memories.map(item => (
                        <div key={item.id} draggable onDragStart={(e) => handleDragStart(e, item)} className="card rounded-xl overflow-hidden flex flex-col" onClick={() => onSelect(item)}>
                            {item.imageData && <img src={item.imageData} className="w-full h-40 object-cover" alt="Memory" />}
                            <div className="p-4 flex flex-col flex-grow">
                                <h3 className="font-semibold text-sm mb-1" style={{wordWrap: 'break-word'}}>{item.title}</h3>
                                <span className="text-xs text-secondary-text mb-2">{formatDate(item.createdAt)}</span>
                                <div className="flex-grow"></div>
                                {item.tags && item.tags.length > 0 && (
                                     <div className="mt-3 flex flex-wrap gap-1">
                                        {item.tags.slice(0, 3).map((tag, index) => (
                                            <span key={index} className="bg-input text-xs font-medium px-2 py-0.5 rounded-full">{tag}</span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </section>
    );
};

const RightSidebar = ({ items, onAddTaskFromCalendar, onToggle, onSelectTodo }) => {
    const [currentDate, setCurrentDate] = useState(new Date());
    const [selectedDate, setSelectedDate] = useState(toISODateString(new Date()));

    const renderCalendarDays = () => {
        const year = currentDate.getFullYear();
        const month = currentDate.getMonth();
        const firstDay = new Date(year, month, 1).getDay();
        const daysInMonth = new Date(year, month + 1, 0).getDate();
        const days = [];

        for (let i = 0; i < firstDay; i++) {
            days.push(<div key={`empty-${i}`}></div>);
        }

        for (let i = 1; i <= daysInMonth; i++) {
            const dayDate = new Date(year, month, i);
            const dateStr = toISODateString(dayDate);
            const isToday = toISODateString(new Date()) === dateStr;
            days.push(
                <div
                    key={i}
                    onClick={() => setSelectedDate(dateStr)}
                    className={`calendar-day ${dateStr === selectedDate ? 'selected' : ''} ${isToday ? 'today' : ''}`}
                >
                    {i}
                </div>
            );
        }
        return days;
    };

    const todosForSelectedDate = items.filter(item => item.type === 'todo' && item.deadline === selectedDate);

    return (
        <aside className="w-64 bg-sidebar p-4 flex flex-col space-y-6">
            <section>
                <h2 className="text-lg font-semibold mb-3">Calendar</h2>
                <div className="bg-surface rounded-lg p-3">
                    <div className="flex justify-between items-center mb-2 px-1">
                        <button onClick={() => setCurrentDate(new Date(currentDate.setMonth(currentDate.getMonth() - 1)))}>&lt;</button>
                        <h3 className="font-semibold text-sm">{currentDate.toLocaleString('default', { month: 'long' })} {currentDate.getFullYear()}</h3>
                        <button onClick={() => setCurrentDate(new Date(currentDate.setMonth(currentDate.getMonth() + 1)))}>&gt;</button>
                    </div>
                    <div className="grid grid-cols-7 gap-2 text-secondary-text text-xs mb-2 text-center">
                        <div>Su</div><div>Mo</div><div>Tu</div><div>We</div><div>Th</div><div>Fr</div><div>Sa</div>
                    </div>
                    <div className="grid grid-cols-7 gap-1">
                        {renderCalendarDays()}
                    </div>
                </div>
                <button onClick={() => onAddTaskFromCalendar(selectedDate)} className="mt-3 w-full bg-surface hover:bg-hover text-primary-text text-sm py-2 rounded-lg transition-colors">Add To-Do</button>
            </section>
            <section>
                <h2 className="text-lg font-semibold mb-3">To-Do</h2>
                <div className="bg-surface rounded-lg p-4 space-y-2">
                    {todosForSelectedDate.length > 0 ? (
                        todosForSelectedDate.map(item => (
                             <div key={item.id} onClick={() => onSelectTodo(item)} className="cursor-pointer">
                                {item.note.split('\n').map((line, index) => {
                                    const match = line.match(/-\s*\[( |x|X)\]\s*(.*)/);
                                    if (!match) return null;
                                    const text = match[2];
                                    const isChecked = match[1].toLowerCase() === 'x';
                                    const checkboxId = `sidebar-todo-${item.id}-${index}`;
                                    return (
                                        <div key={index} className="todo-item flex items-center text-sm">
                                            <input
                                                type="checkbox"
                                                id={checkboxId}
                                                className="mr-2 accent-accent"
                                                checked={isChecked}
                                                onChange={(e) => {
                                                    e.stopPropagation();
                                                    onToggle(item.id, index);
                                                }}
                                            />
                                            <label htmlFor={checkboxId} className="flex items-center cursor-pointer">
                                                <span className={isChecked ? 'text-secondary-text line-through' : ''}>{text}</span>
                                            </label>
                                        </div>
                                    );
                                })}
                            </div>
                        ))
                    ) : (
                        <p className="text-secondary-text text-sm">No tasks for selected date.</p>
                    )}
                </div>
            </section>
        </aside>
    );
};

const Modal = ({ item, onClose, onUpdate, onDelete, onSaveTodo }) => {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [isEditingTitle, setIsEditingTitle] = useState(false);
    const [isEditingSummary, setIsEditingSummary] = useState(false);
    const [title, setTitle] = useState(item.title);
    const [summary, setSummary] = useState(item.summary);
    const [addedTasks, setAddedTasks] = useState([]);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);
    const titleRef = useRef(null);
    const datePickerRef = useRef(null);
    const datePickerButtonRef = useRef(null);

    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === 'Escape') {
                if (isDatePickerOpen) {
                    setIsDatePickerOpen(false);
                } else if (isMenuOpen) {
                    setIsMenuOpen(false);
                } else {
                    onClose();
                }
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isDatePickerOpen, isMenuOpen, onClose]);


    useEffect(() => {
        if (isEditingTitle && titleRef.current) {
            titleRef.current.focus();
            const range = document.createRange();
            range.selectNodeContents(titleRef.current);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
        }
    }, [isEditingTitle]);
    
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (datePickerRef.current && !datePickerRef.current.contains(event.target)) {
                setIsDatePickerOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleTitleSave = () => {
        setIsEditingTitle(false);
        onUpdate({ ...item, title });
    };

    const handleSummarySave = () => {
        setIsEditingSummary(false);
        onUpdate({ ...item, summary });
    };

    const handleAddTask = (task) => {
        onSaveTodo({
            note: `- [ ] ${task.description}`,
            type: 'todo',
            deadline: task.date || null
        }, {
            title: task.description,
            summary: `Task created from '${item.title}'.`,
        });
        setAddedTasks([...addedTasks, task.description]);
    };

    const handleAddSummaryAsTodo = (date) => {
        if (!summary) return;
        onSaveTodo({
            note: `- [ ] ${summary}`,
            type: 'todo',
            deadline: date
        }, {
            title: summary,
            summary: `Task created from '${item.title}'.`
        });
        setIsDatePickerOpen(false);
    };
    
    const copyToClipboard = (textToCopy) => {
        const textArea = document.createElement("textarea");
        textArea.value = textToCopy;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
        document.body.removeChild(textArea);
    };

    if (!item) return null;

    return (
        <div className="fixed inset-0 z-40 fade-in" style={{ backgroundColor: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(5px)' }}>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl rounded-2xl shadow-2xl flex flex-col max-h-[90vh] bg-modal text-base">
                <div className="p-4 flex justify-between items-center border-b border-border-color">
                    <button onClick={onClose} className="text-2xl p-2 rounded-full hover:bg-hover">&larr;</button>
                    <h2
                        ref={titleRef}
                        contentEditable={isEditingTitle}
                        onBlur={handleTitleSave}
                        onInput={(e) => setTitle(e.currentTarget.textContent)}
                        onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleTitleSave(); } }}
                        suppressContentEditableWarning={true}
                        className={`font-semibold text-lg text-center flex-1 mx-4 ${isEditingTitle ? 'focus:outline-1 focus:outline-accent rounded-md' : ''}`}
                    >
                        {item.title}
                    </h2>
                    <div className="relative">
                        <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="text-2xl p-2 rounded-full hover:bg-hover">&hellip;</button>
                        {isMenuOpen && (
                            <div className="absolute right-0 mt-2 w-48 rounded-md shadow-lg z-50 ring-1 ring-border-color bg-surface">
                                <a href="#" onClick={(e) => { e.preventDefault(); setIsEditingTitle(true); setIsMenuOpen(false); }} className="block px-4 py-2 text-base text-primary-text hover:bg-hover rounded-t-md border-b border-border-color">Rename</a>
                                <a href="#" onClick={(e) => { e.preventDefault(); setShowDeleteConfirm(true); setIsMenuOpen(false); }} className="block px-4 py-2 text-base text-red-500 hover:bg-hover rounded-b-md">Delete</a>
                            </div>
                        )}
                    </div>
                </div>
                <div className="overflow-y-auto p-6 space-y-6 text-base modal-content">
                    {item.imageData && (
                        <div className="bg-black rounded-lg flex items-center justify-center p-2 cursor-pointer">
                            <a href={item.imageData} target="_blank" rel="noopener noreferrer">
                                <img src={item.imageData} className="max-h-full max-w-full object-contain rounded-lg" alt="Memory content" />
                            </a>
                        </div>
                    )}
                    <div>
                        <span className="text-sm text-secondary-text">{formatDate(item.createdAt)}</span>
                    </div>
                    {item.tags && item.tags.length > 0 && (
                         <div>
                            <h3 className="font-semibold mb-2">📌 Tags</h3>
                            <div className="flex flex-wrap gap-2">
                                {item.tags.map((tag, index) => (
                                    <span key={index} className="bg-input text-base font-medium px-2.5 py-1 rounded-full">{tag}</span>
                                ))}
                            </div>
                        </div>
                    )}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <h3 className="font-semibold">✨ AI Summary</h3>
                            <div className="flex items-center space-x-2">
                                <button onClick={() => copyToClipboard(summary)} className="p-1 hover:bg-hover rounded-full" title="Copy Summary">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                                </button>
                                {!isEditingSummary ? (
                                    <button onClick={() => setIsEditingSummary(true)} className="p-1 hover:bg-hover rounded-full" title="Edit Summary">
                                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z"></path><path fillRule="evenodd" d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" clipRule="evenodd"></path></svg>
                                    </button>
                                ) : (
                                    <button onClick={handleSummarySave} className="text-base bg-accent px-3 py-1 rounded-md">Save</button>
                                )}
                            </div>
                        </div>
                        <textarea
                            value={summary}
                            onChange={(e) => setSummary(e.target.value)}
                            readOnly={!isEditingSummary}
                            className="w-full bg-input rounded-md p-2 text-base focus:outline-none"
                            rows="5"
                        ></textarea>
                        <div className="mt-2 flex items-center space-x-2 relative" ref={datePickerRef}>
                             <button ref={datePickerButtonRef} onClick={() => setIsDatePickerOpen(prev => !prev)} className="text-base border border-border-color bg-surface hover:bg-hover px-3 py-1 rounded-md transition-colors">Add to To-Do</button>
                             {isDatePickerOpen && (
                                <DatePickerWidget 
                                    anchorRef={datePickerButtonRef}
                                    onSelect={(date) => {
                                        handleAddSummaryAsTodo(date);
                                    }} 
                                    close={() => setIsDatePickerOpen(false)} 
                                />
                             )}
                        </div>
                    </div>

                    {item.highlights && item.highlights.length > 0 && (
                        <div>
                            <h3 className="font-semibold mb-2">💡 Highlights</h3>
                            <div className="space-y-2">
                                {item.highlights.map((task, index) => {
                                    const isAdded = addedTasks.includes(task.description);
                                    return (
                                        <div key={index} className="flex items-center justify-between bg-input p-2 rounded-md text-base">
                                            <div>
                                                <span>{task.description}</span>
                                                {task.date && <span className="ml-2 text-sm bg-surface px-2 py-1 rounded-full text-secondary-text">{formatDueDate(task.date)}</span>}
                                            </div>
                                            <button
                                                onClick={() => handleAddTask(task)}
                                                disabled={isAdded}
                                                className={`p-1 rounded-full transition-colors ${isAdded ? 'text-green-500' : 'hover:bg-hover'}`}
                                                title={isAdded ? 'Added!' : 'Add to To-Do'}
                                            >
                                                {isAdded ? (
                                                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path></svg>
                                                ) : (
                                                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z"></path></svg>
                                                )}
                                            </button>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {item.extractedText && (
                        <div>
                            <div className="flex justify-between items-center mb-1">
                                <h3 className="font-semibold">📝 Extracted Text</h3>
                                <button onClick={() => copyToClipboard(item.extractedText)} className="p-1 hover:bg-hover rounded-full" title="Copy Extracted Text">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                                </button>
                            </div>
                            <pre className="w-full bg-input rounded-md p-2 text-base whitespace-pre-wrap font-sans max-h-40 overflow-y-auto">{item.extractedText}</pre>
                        </div>
                    )}
                </div>
            </div>
            {showDeleteConfirm && (
                <div className="absolute inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: 'rgba(0,0,0,0.5)'}}>
                    <div className="bg-modal rounded-lg p-6 shadow-xl">
                        <h3 className="font-semibold mb-4">Are you sure?</h3>
                        <p className="text-sm text-secondary-text mb-6">This memory will be permanently deleted.</p>
                        <div className="flex justify-end space-x-3">
                            <button onClick={() => setShowDeleteConfirm(false)} className="px-4 py-2 rounded-md bg-surface hover:bg-hover text-sm font-semibold">Cancel</button>
                            <button onClick={() => {onDelete(item.id); setShowDeleteConfirm(false);}} className="px-4 py-2 rounded-md bg-red-600 hover:bg-red-700 text-white text-sm font-semibold">Delete</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

const TodoDetailModal = ({ item, onClose, onUpdate }) => {
    const [taskText, setTaskText] = useState(item ? item.note.match(/-\s*\[( |x|X)\]\s*(.*)/)[2] : '');
    const [deadline, setDeadline] = useState(item ? item.deadline : null);
    const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);
    const datePickerButtonRef = useRef(null);

    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose]);

    const handleSave = () => {
        const updatedNote = item.note.replace(/-\s*\[( |x|X)\]\s*(.*)/, `- [${item.note.includes('[x]') || item.note.includes('[X]') ? 'x' : ' '}] ${taskText}`);
        onUpdate({ ...item, note: updatedNote, deadline });
        onClose();
    };

    if (!item) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(5px)' }}>
            <div className="bg-modal rounded-lg p-6 shadow-xl w-full max-w-md">
                <h3 className="font-semibold mb-4 text-lg">Edit Task</h3>
                <textarea
                    value={taskText}
                    onChange={(e) => setTaskText(e.target.value)}
                    className="w-full bg-input border border-border-color rounded-md p-2 text-sm focus:outline-none focus:ring-1 focus:ring-accent mb-4"
                    rows="3"
                />
                <div className="flex items-center justify-between mb-6">
                    <span className="text-sm font-medium">Deadline:</span>
                    <div className="relative">
                        <button ref={datePickerButtonRef} onClick={() => setIsDatePickerOpen(prev => !prev)} className="text-sm border border-border-color bg-surface hover:bg-hover px-3 py-1 rounded-md transition-colors">
                            {deadline ? formatDueDate(deadline) : 'Set Date'}
                        </button>
                        {isDatePickerOpen && (
                            <DatePickerWidget 
                                anchorRef={datePickerButtonRef}
                                onSelect={(date) => {
                                setDeadline(date);
                                setIsDatePickerOpen(false);
                            }} close={() => setIsDatePickerOpen(false)} />
                        )}
                    </div>
                </div>
                <div className="flex justify-end space-x-3">
                    <button onClick={onClose} className="px-4 py-2 rounded-md bg-surface hover:bg-hover text-sm font-semibold">Cancel</button>
                    <button onClick={handleSave} className="px-4 py-2 rounded-md bg-accent hover:bg-blue-500 text-white text-sm font-semibold">Save</button>
                </div>
            </div>
        </div>
    );
};

export default function App() {
    const [items, setItems] = useState([]);
    const [collections, setCollections] = useState([]);
    const [selectedItem, setSelectedItem] = useState(null);
    const [selectedTodo, setSelectedTodo] = useState(null);
    const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark');
    const [isProcessingScreenshot, setIsProcessingScreenshot] = useState(false);
    const [activePage, setActivePage] = useState('home');
    const addTodoRef = useRef(null);
    const [toast, setToast] = useState(null);
    const [datePickerState, setDatePickerState] = useState({ isOpen: false, itemId: null, anchorRef: null });
    const [selectedCollectionId, setSelectedCollectionId] = useState(null);
    const [collectionMenuState, setCollectionMenuState] = useState({ isOpen: false, collection: null, position: { top: 0, left: 0 } });
    const [isFullScreenNote, setIsFullScreenNote] = useState(false);
    const [fullScreenNoteContent, setFullScreenNoteContent] = useState('');


    useEffect(() => {
        const items = JSON.parse(localStorage.getItem('essentialSpaceItems')||'[]');
        items.forEach(i=>{ if(!('collectionId' in i)) i.collectionId = null });
        localStorage.setItem('essentialSpaceItems', JSON.stringify(items));
    }, []);

    useEffect(() => {
        localStorage.setItem('theme', theme);
    }, [theme]);

    useEffect(() => {
        try {
            const storedItems = JSON.parse(localStorage.getItem('essentialSpaceItems')) || [];
            const storedCollections = JSON.parse(localStorage.getItem('essentialSpaceCollections')) || [];
            setItems(storedItems);
            setCollections(storedCollections);
        } catch (e) {
            console.error("Failed to parse items from localStorage", e);
            setItems([]);
            setCollections([]);
        }
    }, []);

    const saveItems = (newItems) => {
        setItems(newItems);
        localStorage.setItem('essentialSpaceItems', JSON.stringify(newItems));
    };

    const saveCollections = (newCollections) => {
        setCollections(newCollections);
        localStorage.setItem('essentialSpaceCollections', JSON.stringify(newCollections));
    };

    const handleSave = (details, aiData) => {
        const newItem = {
            id: Date.now(),
            ...details,
            ...aiData,
            createdAt: new Date().toISOString(),
            collectionId: null,
        };
        saveItems([...items, newItem]);
    };

    const handleUpdate = (updatedItem) => {
        const newItems = items.map(item => item.id === updatedItem.id ? updatedItem : item);
        saveItems(newItems);
        if(selectedItem && selectedItem.id === updatedItem.id){
            setSelectedItem(updatedItem);
        }
        if(selectedTodo && selectedTodo.id === updatedItem.id){
            setSelectedTodo(updatedItem);
        }
    };

    const handleDelete = (id) => {
        const newItems = items.filter(item => item.id !== id);
        saveItems(newItems);
        setSelectedItem(null);
    };

    const handleToggleTodo = (itemId, lineIndex) => {
        const newItems = items.map(item => {
            if (item.id === itemId && item.type === 'todo') {
                let lines = item.note.split('\n');
                if (lines[lineIndex]) {
                    const isChecked = lines[lineIndex].includes('[x]') || lines[lineIndex].includes('[X]');
                    lines[lineIndex] = isChecked ? lines[lineIndex].replace(/\[x\]/i, '[ ]') : lines[lineIndex].replace(/\[ \]/, '[x]');
                    return { ...item, note: lines.join('\n') };
                }
            }
            return item;
        });
        saveItems(newItems);
    };

    const handlePaste = useCallback(async (event) => {
        const imageItem = Array.from(event.clipboardData.items).find(item => item.type.startsWith('image/'));
        if (imageItem) {
            event.preventDefault();
            setIsProcessingScreenshot(true);
            const file = imageItem.getAsFile();
            if (!file) {
                setIsProcessingScreenshot(false);
                return;
            }

            const reader = new FileReader();
            reader.onload = async (e) => {
                const imageData = e.target.result;
                const aiData = await processContent({ imageData });
                handleSave({ imageData, type: 'memory' }, aiData);
                setIsProcessingScreenshot(false);
            };
            reader.onerror = () => {
                setIsProcessingScreenshot(false);
            }
            reader.readAsDataURL(file);
        }
    }, [items]); 

    useEffect(() => {
        window.addEventListener('paste', handlePaste);
        return () => window.removeEventListener('paste', handlePaste);
    }, [handlePaste]);
    
    const toggleTheme = () => {
        setTheme(prevTheme => prevTheme === 'dark' ? 'light' : 'dark');
    };

    const showToast = (message) => {
        setToast(message);
        setTimeout(() => setToast(null), 3000);
    };

    const handleSaveCollection = (collection) => {
        saveCollections([...collections, collection]);
        showToast("Collection created");
    };

    const handleDeleteCollection = (id) => {
        const newCollections = collections.filter(c => c.id !== id);
        const newItems = items.filter(i => i.collectionId !== id);
        saveCollections(newCollections);
        saveItems(newItems);
        showToast("Collection deleted & memories removed");
    };

    const handleRemoveCollection = (id) => {
        const newCollections = collections.filter(c => c.id !== id);
        const newItems = items.map(i => i.collectionId === id ? { ...i, collectionId: null } : i);
        saveCollections(newCollections);
        saveItems(newItems);
        showToast("Collection removed");
    };

    const handleDropOnCollection = (itemId, collectionId) => {
        const newItems = items.map(i => i.id === parseInt(itemId) ? { ...i, collectionId } : i);
        saveItems(newItems);
        const collection = collections.find(c => c.id === collectionId);
        showToast(`Memory moved to ${collection.name}`);
    };
    
    const handleShowDatePicker = (itemId, anchorRef) => {
        setDatePickerState({ isOpen: true, itemId, anchorRef });
    };

    const handleDateSelect = (date) => {
        const itemToUpdate = items.find(i => i.id === datePickerState.itemId);
        if (itemToUpdate) {
            handleUpdate({ ...itemToUpdate, deadline: date });
            showToast("Deadline set");
        }
        setDatePickerState({ isOpen: false, itemId: null, anchorRef: null });
    };
    
    const handleOpenCollectionMenu = (collection, target) => {
        const rect = target.getBoundingClientRect();
        setCollectionMenuState({ isOpen: true, collection, position: { top: rect.bottom, left: rect.left } });
    };

    const handleCloseCollectionMenu = () => {
        setCollectionMenuState({ isOpen: false, collection: null, position: { top: 0, left: 0 } });
    };

    const handleFullScreenNote = (note) => {
        setFullScreenNoteContent(note);
        setIsFullScreenNote(true);
    };

    return (
        <>
        <style>{`
            :root {
                --accent: #3b82f6;
            }
            [data-theme='dark'] {
                --background: #0a0a0a; --surface: #1a1a1a; --primary-text: #f0f0f0;
                --secondary-text: #a0a0a0; --border-color: #2a2a2a; --sidebar: #000000;
                --input: #2c2c2e; --hover: #252525; --modal: #1c1c1e;
            }
            [data-theme='light'] {
                --background: #f0f2f5; --surface: #ffffff; --primary-text: #1c1c1e;
                --secondary-text: #6e6e73; --border-color: #d1d1d6; --sidebar: #ffffff;
                --input: #e9e9eb; --hover: #f5f5f7; --modal: #ffffff;
            }
            body { 
                font-family: 'Inter', sans-serif; 
                background-color: var(--background); 
                color: var(--primary-text); 
                overflow: hidden; 
                transition: background-color 0.3s, color 0.3s;
            }
            h1, h2, h3, h4, h5, h6, p, div, button, a, span, label, pre {
                color: var(--primary-text);
            }
            input, textarea {
                color: var(--primary-text);
            }
            ::placeholder {
                color: var(--secondary-text);
            }
            main {
                background-color: var(--background);
            }
            .bg-surface { background-color: var(--surface); }
            .bg-sidebar { background-color: var(--sidebar); }
            .bg-input { background-color: var(--input); }
            .bg-hover:hover { background-color: var(--hover); }
            .bg-modal { background-color: var(--modal); }
            .text-primary-text { color: var(--primary-text); }
            .text-secondary-text { color: var(--secondary-text); }
            .border-border-color { border-color: var(--border-color); }
            .card { 
                background-color: var(--surface); 
                transition: transform 0.2s ease-out, box-shadow 0.2s ease-out; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                cursor: pointer;
                height: 294px; /* Fixed height for cards */
            }
            .card .p-4 {
                padding-bottom: 1.5rem;
            }
            [data-theme='dark'] .card { box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
            .card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }
            [data-theme='dark'] .card:hover { box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
            .nav-link { color: var(--primary-text); }
            .nav-link.active { background-color: var(--accent); color: white; }
            .nav-link.active span, .nav-link.active svg { color: white; }
            .nav-link:not(.active):hover { background-color: var(--hover); }
            .calendar-day { text-align: center; padding: 4px; border-radius: 9999px; font-size: 0.75rem; cursor: pointer; transition: background-color 0.2s; color: var(--primary-text); }
            .calendar-day:hover { background-color: var(--hover); }
            .calendar-day.selected { background-color: var(--accent); color: white; }
            .calendar-day.today { box-shadow: inset 0 0 0 1px var(--accent); }
            .processing-spinner {
                width: 24px;
                height: 24px;
                border: 3px solid var(--accent);
                border-bottom-color: transparent;
                border-radius: 50%;
                display: inline-block;
                box-sizing: border-box;
                animation: rotation 1s linear infinite;
            }
            @keyframes rotation {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .line-clamp-2 {
                overflow: hidden;
                display: -webkit-box;
                -webkit-box-orient: vertical;
                -webkit-line-clamp: 2;
            }
            .modal-content::-webkit-scrollbar {
                width: 8px;
            }
            .modal-content::-webkit-scrollbar-track {
                background: var(--surface);
            }
            .modal-content::-webkit-scrollbar-thumb {
                background-color: var(--border-color);
                border-radius: 10px;
                border: 2px solid var(--surface);
            }
            .toast {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background-color: var(--accent);
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                z-index: 100;
            }
            .truncate-2-words {
                overflow: hidden;
                text-overflow: ellipsis;
                display: -webkit-box;
                -webkit-line-clamp: 1;
                -webkit-box-orient: vertical;
                word-break: break-all;
            }
            .collection-item.bg-accent, .collection-item:hover {
                background-color: var(--accent);
            }
            .collection-item.bg-accent span, .collection-item:hover span {
                color: white;
            }
        `}</style>
        <div className="flex h-screen" data-theme={theme}>
            <Sidebar 
                theme={theme} 
                toggleTheme={toggleTheme} 
                activePage={activePage} 
                setActivePage={setActivePage} 
                collections={collections}
                onSaveCollection={handleSaveCollection}
                onOpenCollectionMenu={handleOpenCollectionMenu}
                onDropOnCollection={handleDropOnCollection}
                setSelectedCollectionId={setSelectedCollectionId}
            />
            <main className="flex-1 flex flex-col overflow-y-auto p-6 space-y-6">
                {activePage === 'home' || activePage === 'collection' ? (
                    <>
                        <AddNote onSave={handleSave} isProcessingScreenshot={isProcessingScreenshot} onFullScreen={handleFullScreenNote} />
                        <AddTodo ref={addTodoRef} onSave={handleSave} items={items} onToggle={handleToggleTodo} onSelectTodo={setSelectedTodo} onShowDatePicker={handleShowDatePicker} />
                        <MemoriesFeed items={items} onSelect={setSelectedItem} selectedCollectionId={selectedCollectionId} />
                    </>
                ) : null}
                {activePage === 'todo' && (
                    <TodoPage items={items} onToggle={handleToggleTodo} onSelectTodo={setSelectedTodo} saveItems={saveItems} onShowDatePicker={handleShowDatePicker} />
                )}
            </main>
            <RightSidebar 
                items={items} 
                onToggle={handleToggleTodo}
                onAddTaskFromCalendar={(date) => addTodoRef.current?.setFocusAndDate(date)} 
                onSelectTodo={setSelectedTodo}
            />
            {selectedItem && <Modal 
                item={selectedItem} 
                onClose={() => setSelectedItem(null)} 
                onUpdate={handleUpdate} 
                onDelete={handleDelete}
                onSaveTodo={handleSave}
            />}
            {selectedTodo && <TodoDetailModal
                item={selectedTodo}
                onClose={() => setSelectedTodo(null)}
                onUpdate={handleUpdate}
            />}
            {datePickerState.isOpen && (
                <DatePickerWidget
                    anchorRef={datePickerState.anchorRef}
                    onSelect={handleDateSelect}
                    close={() => setDatePickerState({ isOpen: false, itemId: null, anchorRef: null })}
                />
            )}
            {collectionMenuState.isOpen && (
                <CollectionMenu
                    collection={collectionMenuState.collection}
                    onClose={handleCloseCollectionMenu}
                    onRemove={handleRemoveCollection}
                    onDelete={handleDeleteCollection}
                    position={collectionMenuState.position}
                />
            )}
            {isFullScreenNote && (
                <FullScreenNoteModal
                    initialNote={fullScreenNoteContent}
                    onSave={handleSave}
                    onClose={() => setIsFullScreenNote(false)}
                />
            )}
            {toast && <div className="toast">{toast}</div>}
        </div>
        </>
    );
}

const TodoPage = ({ items, onToggle, onSelectTodo, saveItems, onShowDatePicker }) => {
    const [selectedTodos, setSelectedTodos] = useState([]);
    const todos = items.filter(item => item.type === 'todo');
    const buttonRefs = useRef({});

    const handleSelectAll = () => {
        if (selectedTodos.length === todos.length) {
            setSelectedTodos([]);
        } else {
            setSelectedTodos(todos.map(todo => todo.id));
        }
    };

    const handleDeleteSelected = () => {
        const newItems = items.filter(item => !selectedTodos.includes(item.id));
        saveItems(newItems);
        setSelectedTodos([]);
    };

    const handleSelectTodo = (id) => {
        if (selectedTodos.includes(id)) {
            setSelectedTodos(selectedTodos.filter(todoId => todoId !== id));
        } else {
            setSelectedTodos([...selectedTodos, id]);
        }
    };

    return (
        <div className="bg-surface rounded-lg p-6 h-full flex flex-col">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold">All To-Do Items</h2>
                <div className="flex items-center space-x-4">
                    <button onClick={handleSelectAll} className="text-sm font-medium hover:text-accent">
                        {selectedTodos.length === todos.length ? 'Deselect All' : 'Select All'}
                    </button>
                    <button 
                        onClick={handleDeleteSelected} 
                        disabled={selectedTodos.length === 0}
                        className="text-sm font-medium text-red-500 disabled:text-gray-500 hover:text-red-400 disabled:cursor-not-allowed"
                    >
                        Delete Selected
                    </button>
                </div>
            </div>
            <div className="flex-1 overflow-y-auto">
                {todos.map(item => (
                    <div key={item.id} className="p-2 border-b border-border-color/50">
                        {item.note.split('\n').map((line, index) => {
                            const match = line.match(/-\s*\[( |x|X)\]\s*(.*)/);
                            if (!match) return null;
                            const text = match[2];
                            const isChecked = match[1].toLowerCase() === 'x';
                            return (
                                <div key={index} className="flex items-center justify-between text-sm">
                                    <div className="flex items-center">
                                        <input
                                            type="checkbox"
                                            checked={selectedTodos.includes(item.id)}
                                            onChange={() => handleSelectTodo(item.id)}
                                            className="mr-4 accent-accent h-5 w-5"
                                        />
                                        <span 
                                            onClick={() => onSelectTodo(item)} 
                                            className={`cursor-pointer ${isChecked ? 'text-secondary-text line-through' : ''}`}
                                        >
                                            {text}
                                        </span>
                                    </div>
                                    <div className="relative">
                                        {!item.deadline ? (
                                            <button ref={el => buttonRefs.current[item.id] = el} onClick={(e) => { e.stopPropagation(); onShowDatePicker(item.id, buttonRefs.current[item.id]); }} className="text-xs bg-input px-2 py-1 rounded-full text-white">Add Date</button>
                                        ) : (
                                            <span ref={el => buttonRefs.current[item.id] = el} onClick={(e) => { e.stopPropagation(); onShowDatePicker(item.id, buttonRefs.current[item.id]); }} className="text-xs bg-input px-2 py-1 rounded-full text-secondary-text cursor-pointer">{formatDueDate(item.deadline)}</span>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                ))}
            </div>
        </div>
    );
};
