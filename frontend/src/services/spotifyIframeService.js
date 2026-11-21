/**
 * spotifyIframeService.js
 * =======================
 * Manages singleton loading of Spotify's official iFrame API (v1).
 * https://developer.spotify.com/documentation/embeds/tutorials/using-the-iframe-api
 */

let apiPromise = null;

export function getSpotifyIFrameApi() {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Window is not available'));
  }

  // If already loaded on window
  if (window.SpotifyIframeApi) {
    return Promise.resolve(window.SpotifyIframeApi);
  }

  if (apiPromise) {
    return apiPromise;
  }

  apiPromise = new Promise((resolve, reject) => {
    // Timeout safeguard (5 seconds) so offline or blocked script calls reject cleanly
    const timeout = setTimeout(() => {
      reject(new Error('Spotify iFrame API load timed out'));
    }, 5000);

    const prevReady = window.onSpotifyIframeApiReady;
    window.onSpotifyIframeApiReady = (IFrameAPI) => {
      clearTimeout(timeout);
      window.SpotifyIframeApi = IFrameAPI;
      if (typeof prevReady === 'function') {
        try {
          prevReady(IFrameAPI);
        } catch {
          // ignore external handler error
        }
      }
      resolve(IFrameAPI);
    };

    // Inject script if not already present in DOM
    const existingScript = document.querySelector(
      'script[src="https://open.spotify.com/embed/iframe-api/v1"]'
    );
    if (!existingScript) {
      const script = document.createElement('script');
      script.src = 'https://open.spotify.com/embed/iframe-api/v1';
      script.async = true;
      script.onerror = () => {
        clearTimeout(timeout);
        apiPromise = null;
        reject(new Error('Failed to load Spotify iFrame API script'));
      };
      document.body.appendChild(script);
    }
  });

  return apiPromise;
}
