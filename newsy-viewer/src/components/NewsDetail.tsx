import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Chip, 
  Button, 
  IconButton, 
  Container,
  Divider,
  Avatar
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { NewsItem } from '../types';

interface NewsDetailProps {
  newsItem: NewsItem;
  onClose: () => void;
  onOpenSource: () => void;
}

const NewsDetail: React.FC<NewsDetailProps> = ({ newsItem, onClose, onOpenSource }) => {
  return (
    <Container maxWidth="md" sx={{ mt: 2, mb: 4 }}>
      <Paper 
        elevation={6}
        sx={{ 
          position: 'relative',
          p: { xs: 2, sm: 4 },
          overflow: 'auto',
          maxHeight: '90vh',
          borderRadius: 3,
          backgroundImage: 'linear-gradient(to bottom, #262B35, #1A1D24)',
          boxShadow: '0 10px 40px rgba(0,0,0,0.3)'
        }}
      >
        {/* Close button */}
        <IconButton
          onClick={onClose}
          sx={{ 
            position: 'absolute', 
            top: 16, 
            right: 16,
            backgroundColor: 'rgba(255,255,255,0.1)',
            '&:hover': { backgroundColor: 'rgba(255,255,255,0.2)' },
          }}
          aria-label="close"
        >
          <CloseIcon />
        </IconButton>

        {/* Title and image */}
        <Typography 
          variant="h4" 
          component="h1" 
          sx={{ 
            fontWeight: 700,
            fontSize: { xs: '1.8rem', sm: '2.125rem', md: '2.5rem' },
            mb: 2,
            color: 'white',
            lineHeight: 1.2
          }}
        >
          {newsItem.title}
        </Typography>
        
        {/* Source info */}
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <Avatar 
            sx={{ 
              width: 24, 
              height: 24, 
              mr: 1, 
              bgcolor: 'primary.main',
              fontSize: '0.8rem',
              fontWeight: 'bold'
            }}
          >
            {newsItem.main_source && new URL(newsItem.main_source).hostname.charAt(0).toUpperCase()}
          </Avatar>
          <Typography 
            variant="body2" 
            sx={{ 
              color: 'rgba(255,255,255,0.85)', 
              fontWeight: 500,
              flexGrow: 1
            }}
          >
            {newsItem.main_source && new URL(newsItem.main_source).hostname.replace('www.', '')}
          </Typography>
          <Button 
            startIcon={<OpenInNewIcon />}
            onClick={onOpenSource}
            variant="outlined"
            sx={{ 
              ml: 2,
              borderRadius: 2,
              borderColor: 'rgba(255,255,255,0.3)',
              color: 'rgba(255,255,255,0.85)',
              '&:hover': { borderColor: 'rgba(255,255,255,0.5)' }
            }}
            size="small"
          >
            Open Source
          </Button>
        </Box>
        
        {/* Tags */}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 3 }}>
          {newsItem.tags?.map((tag, index) => (
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
        
        {/* Main image */}
        {newsItem.main_image && (
          <Box 
            component="img" 
            src={newsItem.main_image}
            alt={newsItem.title}
            sx={{ 
              width: '100%', 
              maxHeight: '400px',
              objectFit: 'cover',
              borderRadius: 1,
              mb: 3
            }}
          />
        )}
        
        {/* Summary */}
        <Typography 
          variant="subtitle1" 
          component="div" 
          sx={{ 
            fontWeight: 600,
            mb: 3,
            fontSize: '1.1rem',
            lineHeight: 1.6,
            color: 'white',
            backgroundColor: 'rgba(255,255,255,0.05)',
            p: 2,
            borderRadius: 2,
            borderLeft: '4px solid #4F6BFF'
          }}
        >
          {newsItem.summary}
        </Typography>
        
        <Divider sx={{ my: 3, opacity: 0.2 }} />
        
        {/* Full article content */}
        <Typography variant="body1" component="div" sx={{ mt: 3, whiteSpace: 'pre-wrap', color: 'rgba(255,255,255,0.85)', lineHeight: 1.8, fontSize: '1.05rem' }}>
          {newsItem.body}
        </Typography>
        
        {/* Other images if available */}
        {newsItem.other_images && newsItem.other_images.length > 0 && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>Related Images</Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              {newsItem.other_images.map((img, i) => (
                <Box
                  key={i}
                  component="img"
                  src={img}
                  sx={{
                    width: '150px',
                    height: '100px',
                    objectFit: 'cover',
                    borderRadius: 1
                  }}
                />
              ))}
            </Box>
          </Box>
        )}
        
        {/* Other sources if available */}
        {newsItem.other_sources && newsItem.other_sources.length > 0 && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>Additional Sources</Typography>
            <ul>
              {newsItem.other_sources.map((source, i) => (
                <li key={i}>
                  <Typography 
                    component="a" 
                    href={source} 
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{ color: 'primary.main' }}
                  >
                    {new URL(source).hostname.replace('www.', '')}
                  </Typography>
                </li>
              ))}
            </ul>
          </Box>
        )}
        
        {/* Keyboard shortcuts info */}
        <Box sx={{ mt: 4, bgcolor: 'background.paper', p: 2, borderRadius: 1 }}>
          <Typography variant="caption">
            Keyboard shortcuts: ESC to close, O to open source
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default NewsDetail;