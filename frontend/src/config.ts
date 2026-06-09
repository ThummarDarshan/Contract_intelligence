// When running via Vite dev server (any port except 8000), proxy API calls to FastAPI backend
const port = window.location.port;
export const API_BASE_URL = port && port !== '8000' ? 'http://127.0.0.1:8000' : '';
