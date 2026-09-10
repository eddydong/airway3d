// Bounded, read-only artifact loading. No viewer call can submit work.
export async function cfdAPI(path, body, {timeoutMs = 10000, binary = false} = {}) {
  if(body!==undefined)throw Error('The CFD viewer only reads offline recordings.');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      signal: controller.signal,
      method: 'GET',
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw Error(data.error || `CFD request failed (${response.status})`);
    }
    return await (binary ? response.arrayBuffer() : response.json());
  } catch (error) {
    if (controller.signal.aborted) throw Error('Recording download timed out. Refresh the saved library to retry.');
    if (error instanceof TypeError || error instanceof SyntaxError) throw Error('Recording server unavailable. Start make serve, then refresh the saved library.');
    throw error;
  } finally { clearTimeout(timer); }
}
