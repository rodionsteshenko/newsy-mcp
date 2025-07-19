import React from 'react';
import { Box, Typography, CircularProgress, Container, Grid } from '@mui/material';
import NewsCard from './NewsCard';
import { NewsItem } from '../types';

interface NewsCardListProps {
  newsItems: NewsItem[];
  loading: boolean;
  onCardClick: (newsItem: NewsItem) => void;
  onDismissCard: (newsItem: NewsItem) => void;
}

const NewsCardList: React.FC<NewsCardListProps> = ({ 
  newsItems, 
  loading, 
  onCardClick, 
  onDismissCard 
}) => {
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (newsItems.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', marginTop: 8 }}>
        <Typography variant="h5">No news items available</Typography>
        <Typography color="text.secondary">Check back later for updates</Typography>
      </Box>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: { xs: 2, sm: 4, md: 5 } }}>
      <Typography 
        variant="h4" 
        component="h1" 
        gutterBottom
        sx={{ 
          fontSize: { xs: '1.8rem', sm: '2.125rem', md: '2.5rem' },
          mb: { xs: 2, sm: 3, md: 4 },
          fontWeight: 700,
          textAlign: { xs: 'left', sm: 'left' },
          color: 'white',
          borderLeft: '4px solid #4F6BFF',
          paddingLeft: 2,
          textShadow: '0 2px 4px rgba(0,0,0,0.2)',
        }}
      >
        Latest News
      </Typography>
      
      <Grid container spacing={3}>
        {newsItems.map((item, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <NewsCard
              newsItem={item}
              onClick={() => onCardClick(item)}
              onDismiss={() => onDismissCard(item)}
            />
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default NewsCardList;