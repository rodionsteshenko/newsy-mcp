import React from 'react';
import { Card, CardContent, CardMedia, Typography, Box, Chip, Paper } from '@mui/material';
import { useSwipeable } from 'react-swipeable';
import { NewsItem } from '../types';

interface NewsCardProps {
  newsItem: NewsItem;
  onClick: () => void;
  onDismiss: () => void;
}

const NewsCard: React.FC<NewsCardProps> = ({ newsItem, onClick, onDismiss }) => {
  const swipeHandlers = useSwipeable({
    onSwipedLeft: () => onDismiss(),
    trackMouse: true
  });

  // If there's no main_image, show a default or placeholder image
  const imageUrl = newsItem.main_image || 'https://via.placeholder.com/300x200?text=No+Image';

  return (
    <Card 
      {...swipeHandlers}
      sx={{ 
        marginBottom: 3,
        cursor: 'pointer',
        transition: 'transform 0.2s',
        overflow: 'visible',
        borderRadius: 2,
        boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: '0 12px 28px rgba(0,0,0,0.20)'
        }
      }}
      onClick={onClick}
    >
      {/* Full-width image at the top */}
      <CardMedia
        component="img"
        sx={{ 
          width: '100%', 
          height: 220, 
          objectFit: 'cover'
        }}
        image={imageUrl}
        alt={newsItem.title}
      />
      
      <CardContent sx={{ pt: 2, pb: 2.5 }}>
        {/* Source */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography 
            variant="caption" 
            sx={{ 
              color: 'rgba(255,255,255,0.8)',
              fontWeight: 500,
              letterSpacing: 0.3
            }}
          >
            {newsItem.main_source && new URL(newsItem.main_source).hostname.replace('www.', '')}
          </Typography>
        </Box>
        
        {/* Title */}
        <Typography 
          component="div" 
          variant="h5" 
          gutterBottom 
          sx={{ 
            fontWeight: 600, 
            lineHeight: 1.3,
            mb: 1.5,
            color: 'white'
          }}
        >
          {newsItem.title}
        </Typography>
        
        {/* Summary with better text wrapping */}
        <Typography 
          variant="body1" 
          sx={{ 
            mb: 2,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 3,
            WebkitBoxOrient: 'vertical',
            color: 'rgba(255,255,255,0.9)',
            lineHeight: 1.6,
            fontSize: '1rem'
          }}
        >
          {newsItem.summary}
        </Typography>
        
        {/* Tags */}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.8, mt: 1.5 }}>
          {newsItem.tags?.slice(0, 3).map((tag, index) => (
            <Chip 
              key={index} 
              label={tag} 
              size="small"
              sx={{ 
                borderRadius: '4px',
                backgroundColor: 'rgba(255,255,255,0.1)',
                color: 'rgba(255,255,255,0.95)',
                fontWeight: 500,
                '&:hover': { backgroundColor: 'rgba(255,255,255,0.2)' }
              }} 
            />
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default NewsCard;