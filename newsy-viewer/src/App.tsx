import React, { useState, useEffect } from 'react';
import { CssBaseline, Box, ThemeProvider, createTheme } from '@mui/material';
import NewsCardList from './components/NewsCardList';
import NewsDetail from './components/NewsDetail';
import { NewsItem } from './types';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#121212',
      paper: '#1E1E1E',
    },
    primary: {
      main: '#4F6BFF',
    },
    secondary: {
      main: '#EF6C99',
    },
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", "Roboto", "Helvetica Neue", sans-serif',
    h4: {
      fontWeight: 700,
      letterSpacing: '-0.5px',
    },
    h5: {
      fontWeight: 600,
      letterSpacing: '-0.3px',
    },
    body1: {
      lineHeight: 1.7,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(to bottom, #262B35, #1A1D24)',
          backgroundSize: 'cover',
        },
      },
    },
  },
});

function App() {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<NewsItem | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    // In a real app, this would be fetched from an API
    const fetchNewsData = async () => {
      try {
        const response = await fetch('/news.jsonl');
        const text = await response.text();
        
        // Parse JSONL (each line is a separate JSON object)
        const items = text
          .split('\n')
          .filter(line => line.trim())
          .map(line => JSON.parse(line));
        
        setNewsItems(items);
        setLoading(false);
      } catch (error) {
        console.error('Error loading news data:', error);
        setLoading(false);
      }
    };

    fetchNewsData();
  }, []);

  const handleCardClick = (item: NewsItem) => {
    setSelectedItem(item);
  };

  const handleDetailClose = () => {
    setSelectedItem(null);
  };

  const handleDismissCard = (itemToRemove: NewsItem) => {
    setNewsItems(newsItems.filter(item => item.title !== itemToRemove.title));
  };
  
  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape' && selectedItem) {
      handleDetailClose();
    } else if (event.key === 'o' && selectedItem) {
      window.open(selectedItem.main_source, '_blank');
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedItem]);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        minHeight: '100vh',
        padding: { xs: 0, sm: 2, md: 3 },
        backgroundColor: '#121212',
        backgroundImage: 'linear-gradient(to bottom, #15171E, #0C0E14)'
      }}>
        {!selectedItem ? (
          <NewsCardList 
            newsItems={newsItems} 
            loading={loading} 
            onCardClick={handleCardClick}
            onDismissCard={handleDismissCard}
          />
        ) : (
          <NewsDetail 
            newsItem={selectedItem} 
            onClose={handleDetailClose} 
            onOpenSource={() => window.open(selectedItem.main_source, '_blank')}
          />
        )}
      </Box>
    </ThemeProvider>
  );
}

export default App;