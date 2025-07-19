export interface NewsItem {
  title: string;
  summary: string;
  tags: string[];
  main_image?: string;
  other_images?: string[];
  body: string;
  main_source: string;
  other_sources?: string[];
}