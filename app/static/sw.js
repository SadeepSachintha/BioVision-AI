/**
 * BioVision AI — Offline Rural Field Service Worker (PWA)
 * Caches UI shell, styling, scripts, sample leaves, and agronomic knowledge base
 * to ensure 100% operational capability in zero-connectivity rural dead zones.
 */

const CACHE_NAME = 'biovision-offline-v1';

const ASSETS_TO_CACHE = [
  '/ui',
  '/ui/',
  '/ui/style.css',
  '/ui/app.js',
  '/ui/disease_db.json',
  '/sample_leaves/tomato_late_blight.jpg',
  '/sample_leaves/corn_common_rust.jpg',
  '/sample_leaves/potato_early_blight.jpg',
  '/sample_leaves/apple_healthy.jpg'
];

// Install Event: Cache Core Offline Assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[BioVision PWA] Pre-caching offline rural field shell assets...');
      return cache.addAll(ASSETS_TO_CACHE);
    }).then(() => self.skipWaiting())
  );
});

// Activate Event: Clean up outdated caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (name !== CACHE_NAME) {
            console.log('[BioVision PWA] Clearing stale cache:', name);
            return caches.delete(name);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Event: Network-first with Cache fallback, or Cache-first for static assets
self.addEventListener('fetch', (event) => {
  const requestUrl = new URL(event.request.url);

  // If requesting static UI or disease knowledge base, serve from cache with network update
  if (requestUrl.pathname.startsWith('/ui') || requestUrl.pathname.startsWith('/sample_leaves')) {
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        if (cachedResponse) {
          // Fetch fresh copy in background if online
          fetch(event.request).then((networkResponse) => {
            if (networkResponse && networkResponse.status === 200) {
              caches.open(CACHE_NAME).then((cache) => cache.put(event.request, networkResponse));
            }
          }).catch(() => {/* offline; cached response already served */});
          return cachedResponse;
        }
        return fetch(event.request);
      })
    );
    return;
  }

  // Standard network request with fallback
  event.respondWith(
    fetch(event.request).catch(() => {
      return caches.match(event.request);
    })
  );
});
