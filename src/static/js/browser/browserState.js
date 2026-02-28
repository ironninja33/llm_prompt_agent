/**
 * Browser shared state.
 */

const BrowserState = {
    currentPath: '',
    items: [],
    offset: 0,
    limit: 50,
    hasMore: false,
    isLoading: false,
    pollTimer: null,
    pollTimestamp: Date.now() / 1000,
    searchMode: 'keyword',  // 'keyword' or 'embedding'
    searchQuery: '',
    isSearchActive: false,
    thumbnailSize: 'medium',
};
