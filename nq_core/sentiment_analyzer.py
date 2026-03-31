import re

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
    
    # Optional runtime download if missing (often fails in restricted environments)
    # nltk.download('vader_lexicon', quiet=True) 
    
except ImportError:
    NLTK_AVAILABLE = False
    
class SentimentAnalyzer:
    """
    NLP Sentiment Analysis Tool for Market Context.
    Evaluates news headlines or FOMC statements to yield a continuous [-1.0 to 1.0] 
    bull/bear sentiment score.
    
    Features built-in Graceful Degradation: If NLTK or the Lexicon fails to load,
    the score rigidly reverts to 0.0 (Neutral), bypassing the filter seamlessly.
    """
    
    def __init__(self, use_sentiment=False):
        self.use_sentiment = use_sentiment
        self.analyzer = None
        
        if self.use_sentiment and NLTK_AVAILABLE:
            try:
                self.analyzer = SentimentIntensityAnalyzer()
                self._is_ready = True
            except Exception as e:
                # Vader Lexicon not found or another NLTK issue
                print(f"[Sentiment Engine] Warning: Failed to load VADER analyzer -> {e}")
                self._is_ready = False
        else:
            self._is_ready = False
            
    def toggle_sentiment(self, state: bool):
        self.use_sentiment = state
        if self.use_sentiment and NLTK_AVAILABLE and not self._is_ready:
            try:
                self.analyzer = SentimentIntensityAnalyzer()
                self._is_ready = True
            except:
                pass

    def evaluate_text(self, text_blob: str) -> float:
        """
        Ingests a string of news headlines and scores it using VADER.
        Returns a float between -1.0 (Extreme Fear/Bearish) and 1.0 (Extreme Greed/Bullish).
        Returns 0.0 on any failure or if toggled off.
        """
        if not self.use_sentiment or not self._is_ready or not text_blob:
            return 0.0
            
        # Clean text
        text_blob = str(text_blob).lower()
        text_blob = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text_blob)
        
        try:
            scores = self.analyzer.polarity_scores(text_blob)
            
            # VADER returns {'neg': X, 'neu': Y, 'pos': Z, 'compound': C}
            compound_score = scores.get('compound', 0.0)
            
            return compound_score
            
        except Exception:
            # Absolute fallback to neutral on any parsing crash
            return 0.0
