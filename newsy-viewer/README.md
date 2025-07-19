# Newsy Viewer

A React application for viewing news articles from the Newsy RSS system. This viewer displays news articles in a card-based interface and provides detailed views with keyboard shortcuts and swipe interactions.

## Features

- Card-based news article list
- Detailed article view with full content
- Keyboard shortcuts:
  - ESC to close detailed view
  - O to open source article
- Swipe interactions:
  - Swipe left to dismiss cards
- Responsive design for various screen sizes
- Dark mode support

## Prerequisites

- Node.js (v14+)
- npm or yarn

## Installation

1. Clone or download this repository
2. Navigate to the project directory:

```bash
cd newsy-viewer
```

3. Install dependencies:

```bash
npm install
# or
yarn
```

## Development

To start the development server:

```bash
npm run start
# or
yarn start
```

This will start the development server on [http://localhost:5173](http://localhost:5173).

## Building for Production

To build the app for production:

```bash
npm run build
# or
yarn build
```

The build output will be in the `dist` directory.

## Data Format

The viewer expects news data in JSONL format (JSON Lines) where each line is a separate JSON object with the following structure:

```json
{
  "title": "Article headline/title",
  "summary": "Brief summary of the article content",
  "tags": ["array", "of", "relevant", "topic", "tags"],
  "main_image": "URL or path to the primary image (optional)",
  "other_images": ["array", "of", "supplemental", "image", "URLs"],
  "body": "Full processed article content",
  "main_source": "Primary source URL",
  "other_sources": ["array", "of", "additional", "source", "URLs"]
}
```

## Integration with Newsy

This viewer is designed to work with the output from the Newsy RSS system. To use it:

1. Ensure your Newsy system is generating the `news.jsonl` file
2. Copy the `news.jsonl` file to the `public` directory of this project
3. Start or build the viewer as described above

## Technologies Used

- React
- TypeScript
- Material UI
- React Swipeable
- Vite