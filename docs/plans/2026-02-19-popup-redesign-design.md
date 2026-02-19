# NoFishing Popup Redesign Design Document

**Date**: 2026-02-19
**Status**: Approved
**Author**: Design Review

## Overview

Redesign the Chrome extension popup with a modern tabbed interface, API token authentication, whitelist/blacklist quick actions, detection history, and quick settings toggles.

## Goals

1. **API Token Authentication** - Secure API access with token-based auth
2. **Quick Actions** - One-click whitelist/blacklist for current site
3. **Detection History** - Recent scans with timestamps and details
4. **Quick Settings** - Toggle switches for key features
5. **Visual Refresh** - Modern layout, icons, and animations

## Architecture

### Layout: Tabbed Interface

Three-tab layout with persistent header:
- **首页 (Home)** - Current site info + quick actions
- **历史 (History)** - Detection history with timestamps
- **设置 (Settings)** - Quick toggles + API configuration

**Popup Size**: 380px × 550px (slightly taller for history)

### Component Structure

```
popup.html
├── Header (persistent)
│   ├── Logo + Title
│   ├── API Status Indicator
│   └── Token Status Indicator
├── Tab Navigation
│   ├── Home Tab
│   ├── History Tab
│   └── Settings Tab
├── Tab Content Panels
│   ├── #home-panel
│   ├── #history-panel
│   └── #settings-panel
└── Login Modal (overlay)
```

## Tab Designs

### 首页 (Home) Tab

**Components**:
1. **Current Site Section**
   - URL display (truncated)
   - Status badge (Safe/Suspicious/Phishing)
   - Confidence percentage
   - "Re-scan" button

2. **Quick Actions**
   - "Add to Whitelist" button
   - "Add to Blacklist" button
   - Both call backend API endpoints

3. **Today's Stats**
   - Scanned count
   - Blocked count

4. **Quick Check**
   - URL input field
   - Detect button

### 📋 历史 (History) Tab

**Components**:
1. **Filter Bar**
   - Dropdown: All / Safe / Suspicious / Phishing
   - "Clear History" button

2. **History List**
   - Each entry shows:
     - URL (truncated)
     - Risk level indicator icon
     - Risk level text
     - Confidence percentage
     - Relative timestamp (e.g., "2 minutes ago")
     - "View Details" button

3. **Details Modal**
   - Full URL
   - Complete detection results
   - Absolute timestamp
   - Feature breakdown (if available)

**Storage**:
- Max 200 entries in `chrome.storage.local`
- Auto-remove entries older than 30 days
- Key: `detectionHistory` (array)

### ⚙️ 设置 (Settings) Tab

**Components**:
1. **Quick Toggles**
   - Auto-block phishing sites (switch)
   - Show notifications (switch)
   - Auto-scan (switch)
   - Sensitivity dropdown (Low/Medium/High)

2. **API Configuration**
   - Token display (masked)
   - Login status
   - Logout button
   - Update token button

3. **Maintenance**
   - Clear detection cache button
   - Clear history button

## API Token Authentication

### Flow

1. **Initial Load**
   - Check if token exists in `chrome.storage.local`
   - If no token → Show login modal
   - If token exists → Validate via `/api/v1/auth/verify`

2. **Login Modal**
   - Input field for API token
   - Link to dashboard to get token
   - Login button validates token
   - Cancel button closes modal

3. **Token Usage**
   - All API calls include `Authorization: Bearer <token>` header
   - 401 responses trigger re-authentication
   - Logout clears token from storage

### Storage Keys

```javascript
chrome.storage.local({
  apiToken: string,           // JWT token
  tokenExpiry: number,        // Unix timestamp
  detectionHistory: array,    // History entries
  settings: {
    autoBlock: boolean,
    showNotifications: boolean,
    autoScan: boolean,
    sensitivity: 'low'|'medium'|'high'
  }
})
```

## API Integration

### New/Used Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/verify` | Validate token |
| POST | `/api/v1/detect` | Detect URL (with auth) |
| POST | `/api/v1/whitelist` | Add to whitelist |
| POST | `/api/v1/blacklist` | Add to blacklist |
| GET | `/api/v1/whitelist` | Get whitelist entries |
| GET | `/api/v1/blacklist` | Get blacklist entries |
| DELETE | `/api/v1/whitelist/{id}` | Remove from whitelist |
| DELETE | `/api/v1/blacklist/{id}` | Remove from blacklist |

## Visual Design System

### Colors

```css
--primary: #3B82F6;      /* Modern blue */
--success: #10B981;      /* Emerald green */
--warning: #F59E0B;      /* Amber */
--danger: #EF4444;       /* Modern red */
--bg-primary: #FFFFFF;   /* White */
--bg-secondary: #F8FAFC; /* Light gray */
--bg-tertiary: #F1F5F9;  /* Slightly darker */
--text-primary: #1E293B; /* Dark slate */
--text-secondary: #64748B; /* Muted slate */
--border: #E2E8F0;       /* Light border */
```

### Typography

- Font family: System UI / Inter / Segoe UI
- Headers: 600 weight, 14-16px
- Body: 400 weight, 13px
- URLs: Monospace, 12px

### Icons

Embedded SVG icons (Lucide-style):
- Navigation: `home`, `history`, `settings`
- Actions: `shield-check` (whitelist), `shield-x` (blacklist)
- Status: `check-circle`, `alert-triangle`, `x-circle`
- Settings: `bell`, `shield`, `globe`

### Animations

| Element | Duration | Effect |
|---------|----------|--------|
| Tab switch | 200ms | Fade + slide |
| Toggle switch | 150ms | Bounce |
| Status badge | 300ms | Scale on load |
| History entry | 150ms | Slide-in staggered |
| Loading spinner | Continuous | Smooth rotation |

### Component Styling

- **Cards**: 8px rounded, subtle shadow
- **Buttons**: 6px rounded, hover lift effect
- **Toggles**: iOS-style pill shape
- **Inputs**: 4px rounded, focus ring

## Data Flow

### Detection Flow

1. User navigates to URL
2. Background worker intercepts
3. Check local cache first
4. If not cached → API call with token
5. Store result in history
6. Update UI if popup is open
7. Handle phishing based on settings

### History Flow

1. Detection completes
2. Create history entry object
3. Push to `detectionHistory` array
4. Trim to max 200 entries
5. Remove entries older than 30 days
6. Save to `chrome.storage.local`

### Settings Flow

1. User toggles setting
2. Update `settings` object in storage
3. Send message to background worker
4. Background worker updates behavior
5. Show confirmation toast

## Error Handling

| Error | Action |
|-------|--------|
| Invalid token | Show login modal |
| Network failure | Show toast, use cached result |
| 401 Unauthorized | Clear token, show login modal |
| 403 Forbidden | Show "insufficient permissions" toast |
| 429 Rate limit | Show "rate limited" toast, retry after delay |

## File Structure

```
nofishing-extension/
├── src/
│   ├── popup/
│   │   ├── popup.html          # Redesigned structure
│   │   ├── popup.css           # Updated styles
│   │   ├── popup.js            # Main logic
│   │   ├── tabs/
│   │   │   ├── home-tab.js     # Home tab logic
│   │   │   ├── history-tab.js  # History tab logic
│   │   │   └── settings-tab.js # Settings tab logic
│   │   └── components/
│   │       ├── login-modal.js  # Login modal
│   │       └── toast.js        # Toast notifications
│   ├── background/
│   │   └── background.js       # Updated with token auth
│   └── utils/
│       ├── api.js              # API client with token
│       └── storage.js          # Storage helpers
```

## Success Criteria

- [ ] Login modal appears when no token is stored
- [ ] Token is validated on extension load
- [ ] Whitelist/Blacklist buttons work from home tab
- [ ] History shows recent detections with timestamps
- [ ] Settings toggles persist and update behavior
- [ ] Visual design matches mockups
- [ ] All animations are smooth (60fps)
- [ ] Works with existing backend API

## Implementation Notes

1. **Backward Compatibility**: Existing users without tokens will see login modal on first update
2. **Token Refresh**: If using JWT tokens, implement refresh logic
3. **Offline Mode**: Cache last known detection results for offline use
4. **Performance**: Keep history trimmed to avoid storage bloat
