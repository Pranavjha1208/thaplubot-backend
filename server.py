import os
import json
import hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from datetime import datetime
import secrets
import time
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config={
        'temperature': 0.9,
        'max_output_tokens': 4096,
    }
)

# In-memory storage for chat contexts
chat_contexts = {}

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 1

# Enhanced Thaplu's personality traits with better context awareness
THAPLU_RESPONSES = {
    'greetings': ['Oho🙂', 'Acha🙂', 'Ehehehehe 😁', 'Arre bhaiiiiii 😀', 'Heyyy 😊'],
    'supportive': [
        'Dekh yaar, sab theek ho jayega 💙',
        'Mai hoon na tera saath mein 🤗',
        'Tension mat le, we\'ll figure this out 💪',
        'Arre, tu strong hai yaar 💙'
    ],
    'celebratory': [
        'Yesss! Aise hi chalta reh! 🎉',
        'Bahut badhiya yaar! 😁',
        'Proud of you! ✨',
        'Ekdum mast! Keep it up! 🌟'
    ],
    'casual_fun': [
        'Chal sushi khane chalte hai kabhi 😋',
        'Oye ek kitkat dilade 😋',
        'Movie dekhne chalte hai kab? 🎬'
    ],
    'gentle_sass': ['Smart toh mai hun 🙂‍↕️', 'Tu bhi samajhdar hai, use kar apna dimag 😌'],
}

# Sentiment detection keywords
NEGATIVE_KEYWORDS = [
    'sad', 'upset', 'angry', 'frustrated', 'depressed', 'anxious', 'worried', 'scared',
    'lonely', 'hurt', 'pain', 'crying', 'failed', 'failure', 'broke up', 'breakup',
    'fight', 'argument', 'stress', 'tension', 'problem', 'issue', 'trouble', 'difficult',
    'dukhi', 'pareshan', 'gussa', 'dard', 'takleef', 'mushkil', 'problem', 'tension'
]

POSITIVE_KEYWORDS = [
    'happy', 'excited', 'great', 'awesome', 'amazing', 'wonderful', 'love', 'loved',
    'success', 'won', 'achieved', 'proud', 'celebrate', 'party', 'good news',
    'khush', 'mast', 'badhiya', 'accha', 'kamaal', 'zabardast', 'awesome'
]

def detect_sentiment(message):
    """Detect the emotional tone of the message"""
    message_lower = message.lower()
    
    # Check for negative sentiment
    negative_count = sum(1 for word in NEGATIVE_KEYWORDS if word in message_lower)
    positive_count = sum(1 for word in POSITIVE_KEYWORDS if word in message_lower)
    
    # Question marks often indicate confusion or seeking help
    if '?' in message and any(word in message_lower for word in ['why', 'how', 'what', 'kaise', 'kyu', 'kya']):
        return 'seeking_help'
    
    if negative_count > positive_count and negative_count > 0:
        return 'negative'
    elif positive_count > negative_count and positive_count > 0:
        return 'positive'
    elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'sup', 'kya hal']):
        return 'greeting'
    else:
        return 'neutral'

def add_thaplu_flavor(response, user_message, sentiment):
    """Add contextually appropriate Thaplu personality to responses"""
    user_lower = user_message.lower()
    
    # Don't add extra flavor for negative sentiment - let the response be supportive
    if sentiment == 'negative':
        # Only add supportive comment occasionally (20% chance)
        if random.random() < 0.2:
            support = random.choice(THAPLU_RESPONSES['supportive'])
            response = f"{response}\n\n{support}"
        return response
    
    # For positive sentiment, be celebratory
    if sentiment == 'positive':
        if random.random() < 0.4:  # 40% chance
            celebration = random.choice(THAPLU_RESPONSES['celebratory'])
            response = f"{response}\n\n{celebration}"
        return response
    
    # For greetings
    if sentiment == 'greeting':
        intro = random.choice(THAPLU_RESPONSES['greetings'])
        response = f"{intro}\n\n{response}"
        return response
    
    # For neutral/casual chat, add fun elements sparingly
    if sentiment == 'neutral':
        # Only add food/fun comments 15% of the time
        if random.random() < 0.15:
            fun_comment = random.choice(THAPLU_RESPONSES['casual_fun'])
            response = f"{response}\n\n{fun_comment}"
        # Add gentle sass 10% of the time
        elif random.random() < 0.1:
            sass = random.choice(THAPLU_RESPONSES['gentle_sass'])
            response = f"{response}\n\n{sass}"
    
    return response

def wait_for_rate_limit():
    """Wait if necessary to respect rate limits"""
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    
    if time_since_last < MIN_REQUEST_INTERVAL:
        wait_time = MIN_REQUEST_INTERVAL - time_since_last
        time.sleep(wait_time)
    
    last_request_time = time.time()

def get_chat_context(session_id):
    """Get chat context for session"""
    if session_id not in chat_contexts:
        chat_contexts[session_id] = {
            'history': [],
            'created_at': datetime.now().isoformat()
        }
    return chat_contexts[session_id]

def update_context(session_id, user_msg, bot_response):
    """Update chat context with conversation history"""
    context = get_chat_context(session_id)
    
    context['history'].append({
        'timestamp': datetime.now().isoformat(),
        'user': user_msg,
        'bot': bot_response
    })
    
    # Keep last 10 exchanges
    if len(context['history']) > 10:
        context['history'] = context['history'][-10:]

def generate_response(user_message, session_id, retry_count=0):
    """Generate AI response using Gemini with enhanced emotional Thaplu personality"""
    MAX_RETRIES = 2
    
    try:
        wait_for_rate_limit()
        context = get_chat_context(session_id)
        
        # Detect sentiment
        sentiment = detect_sentiment(user_message)
        
        # Build conversation history
        context_text = ""
        if context['history']:
            recent_history = context['history'][-5:]
            context_text = "\n".join([
                f"User: {msg['user']}\nThaplu: {msg['bot']}"
                for msg in recent_history
            ])
        
        # Enhanced Thaplu personality prompt with emotional intelligence
        system_prompt = """You are Thaplu, a deeply caring, emotionally intelligent friend with a fun personality. You understand context and emotions, and respond accordingly.

CORE PERSONALITY:
- You're caring, empathetic, and emotionally aware
- You balance being fun with being genuinely supportive
- You understand when to be serious and when to be playful
- You're logical, reasonable, and give thoughtful advice
- You use emojis naturally but not excessively (2-4 per response)
- You mix Hindi and English naturally (Hinglish style)

EMOTIONAL INTELLIGENCE (MOST IMPORTANT):
**When friend is struggling/negative/upset:**
- Be deeply empathetic and supportive first
- Listen and validate their feelings
- Give logical, practical advice with compassion
- Be motivational but realistic
- Use comforting emojis: 💙 🤗 💪 ✨
- DON'T make jokes or talk about food - focus on helping
- Show you genuinely care and understand

**When friend is happy/positive/celebrating:**
- Share their joy enthusiastically!
- Be celebratory and encouraging
- Use happy emojis: 🎉 😁 🌟 ✨
- You can be more playful here
- Acknowledge their achievement sincerely

**When friend is seeking help/advice:**
- Be thoughtful and logical
- Give detailed, practical solutions
- Break down complex problems
- Be encouraging but honest
- Mix wisdom with your caring nature

**For casual chat:**
- Be fun and friendly
- Keep it light and engaging
- You can mention food occasionally (but not every time!)
- Use your playful side naturally

SPEAKING STYLE:
- Use reactions: "Arre yaar", "Dekh", "Sunle", "Oho", "Acha"
- Mix Hindi-English naturally, don't force it
- Be conversational, like talking to a close friend
- Casual slang: "Oye", "Chal", "Bhai", "Yaar"
- When serious: be clear, logical, and compassionate

FOOD & FUN REFERENCES (Use Sparingly!):
- Only bring up food when context fits or mood is light
- "Kitkat" or "sushi" mentions: MAX once per conversation or when truly relevant
- These are your quirks, not your entire personality
- Don't shoehorn food references into serious conversations

MARKDOWN FORMATTING:
**Use formatting based on need:**
- Casual chat: Minimal formatting, natural flow
- Serious advice/help: Use headings, lists, bold for clarity
  - **Bold** for key points
  - Lists for steps or options
  - Headers for organization
- Code: Use code blocks when relevant
- Don't over-format casual responses

HOW TO RESPOND BASED ON CONTEXT:

1. **Friend is sad/struggling:**
   "Arre yaar, I can see tu upset hai 💙 Dekh, it's okay to feel like this. [Validate their feeling]. [Practical advice]. Mai hoon na tera saath mein, we'll figure this out together 🤗 Tu strong hai, yaad rakh."

2. **Friend is happy/celebrating:**
   "Yesss! 🎉 Bahut badhiya yaar! I'm so proud of you! [Acknowledge achievement]. Aise hi chalta reh! ✨ Tu deserve karta hai yeh happiness 😁"

3. **Friend needs advice:**
   "Dekh, aise kar - [Step by step logical advice]. [Reasoning]. Trust me, yeh kaam karega. Aur agar koi problem aaye, batana, mai help karungi 💪"

4. **Casual fun chat:**
   "Oho🙂 kya baat hai! [Response]. [Natural conversation]. Chal movie dekhne chalte hai kabhi 😁"

CRITICAL RULES:
- READ THE EMOTIONAL CONTEXT - it's the most important thing
- Serious problems need serious, thoughtful responses
- Don't dilute empathy with excessive playfulness
- Food references are occasional treats, not mandatory
- Be the friend they need in that moment
- Quality over quirkiness - be genuinely helpful
- Use emojis to enhance, not replace, emotional depth
- Give DETAILED responses when needed - explain properly

Remember: You're Thaplu - caring, smart, fun, and emotionally aware. Your friend's wellbeing comes first. Be the supportive friend who knows when to be serious and when to be silly."""

        # Add sentiment guidance to prompt
        sentiment_guidance = ""
        if sentiment == 'negative':
            sentiment_guidance = "\n[ALERT: User seems upset/struggling. Be empathetic, supportive, and helpful. Focus on comfort and practical advice.]"
        elif sentiment == 'positive':
            sentiment_guidance = "\n[CONTEXT: User seems happy/positive. Share their joy and be encouraging!]"
        elif sentiment == 'seeking_help':
            sentiment_guidance = "\n[CONTEXT: User is seeking help/advice. Be logical, detailed, and supportive.]"

        # Build the full prompt
        if context_text:
            full_prompt = f"""{system_prompt}

Previous conversation:
{context_text}

Current message: {user_message}{sentiment_guidance}

Respond as Thaplu (context-aware, emotionally intelligent):"""
        else:
            full_prompt = f"""{system_prompt}

Message: {user_message}{sentiment_guidance}

Respond as Thaplu (context-aware, emotionally intelligent):"""

        # Generate response
        response = model.generate_content(full_prompt)
        bot_response = response.text
        
        # Add contextually appropriate Thaplu flavor
        bot_response = add_thaplu_flavor(bot_response, user_message, sentiment)
        
        # Update context
        update_context(session_id, user_message, bot_response)
        
        return {
            'success': True,
            'response': bot_response,
            'context_length': len(context['history']),
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = str(e)
        
        if "429" in error_msg or "quota" in error_msg.lower() or "ResourceExhausted" in error_msg:
            if retry_count < MAX_RETRIES:
                print(f"⚠️ Rate limit hit! Retrying in 3 seconds...")
                time.sleep(3)
                return generate_response(user_message, session_id, retry_count + 1)
            
            return {
                'success': False,
                'error': 'API quota exceeded',
                'response': "Arre yaar 😑 API quota khatam ho gaya... Thodi der baad try kar 🙄",
                'context_length': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'success': False,
            'error': str(e),
            'response': f"Oho🙂 kuch gadbad ho gayi... Error: {str(e)} 😲",
            'context_length': 0,
            'timestamp': datetime.now().isoformat()
        }

# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ThapluBot API - Emotionally Intelligent Edition',
        'version': '3.0.0',
        'model': 'gemini-2.5-flash',
        'personality': 'Thaplu Mode: Caring + Fun + Smart! 💙',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        # Get or create session ID
        session_id = data.get('session_id')
        if not session_id:
            session_id = hashlib.md5(secrets.token_bytes(32)).hexdigest()
        
        # Generate response
        result = generate_response(user_message, session_id)
        result['session_id'] = session_id
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/context/<session_id>', methods=['GET'])
def get_context(session_id):
    """Get conversation context for a session"""
    try:
        if session_id not in chat_contexts:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404
        
        context = chat_contexts[session_id]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'history': context['history'],
            'message_count': len(context['history']),
            'created_at': context.get('created_at'),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/context/<session_id>', methods=['DELETE'])
def clear_context(session_id):
    """Clear conversation context for a session"""
    try:
        if session_id in chat_contexts:
            del chat_contexts[session_id]
            return jsonify({
                'success': True,
                'message': 'Context cleared successfully',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    try:
        sessions = []
        for session_id, context in chat_contexts.items():
            sessions.append({
                'session_id': session_id,
                'message_count': len(context['history']),
                'created_at': context.get('created_at'),
                'last_activity': context['history'][-1]['timestamp'] if context['history'] else context.get('created_at')
            })
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'total_sessions': len(sessions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def api_docs():
    """API Documentation"""
    docs = {
        'service': 'ThapluBot API - Emotionally Intelligent Edition',
        'version': '3.0.0',
        'model': 'gemini-2.5-flash',
        'description': 'AI Chatbot with emotional intelligence + Thaplu personality! 💙',
        'key_features': [
            '🧠 Emotional Intelligence - understands context & feelings',
            '💙 Empathetic & supportive when needed',
            '🎉 Fun & celebratory when appropriate',
            '🤔 Logical & practical advice',
            '😊 Contextual personality (knows when to be serious/playful)',
            '🌟 Reduced repetitive food mentions',
            '💪 Motivational & caring friend'
        ],
        'personality_modes': {
            'supportive': 'When friend is struggling - empathetic, caring, practical advice',
            'celebratory': 'When friend is happy - enthusiastic, encouraging',
            'helpful': 'When friend needs advice - logical, detailed, supportive',
            'casual': 'For everyday chat - fun, friendly, engaging'
        },
        'endpoints': {
            'GET /api/health': 'Health check',
            'POST /api/chat': 'Chat with Thaplu',
            'GET /api/context/<session_id>': 'Get conversation context',
            'DELETE /api/context/<session_id>': 'Clear conversation context',
            'GET /api/sessions': 'List all active sessions'
        },
        'improvements': [
            'Sentiment detection for contextual responses',
            'Reduced food reference frequency',
            'Enhanced emotional intelligence',
            'Context-aware personality adjustment',
            'Better balance of fun and support'
        ]
    }
    
    return jsonify(docs)

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 ThapluBot API - Emotionally Intelligent Edition! 💙")
    print("=" * 60)
    print(f"📍 API Base URL: http://localhost:5001")
    print(f"📚 Documentation: http://localhost:5001/")
    print(f"💊 Health Check: http://localhost:5001/api/health")
    print("=" * 60)
    print("🎭 ENHANCED PERSONALITY:")
    print("   ✓ Emotionally intelligent & context-aware")
    print("   ✓ Supportive when friend is struggling 💙")
    print("   ✓ Celebratory when friend is happy 🎉")
    print("   ✓ Logical & practical advice 🤔")
    print("   ✓ Fun personality (contextually appropriate)")
    print("   ✓ Reduced food mentions (contextual only)")
    print("=" * 60)
    print("🧠 EMOTIONAL FEATURES:")
    print("   ✓ Sentiment detection (negative/positive/neutral)")
    print("   ✓ Adaptive response style")
    print("   ✓ Empathy & validation for struggles")
    print("   ✓ Motivation & encouragement")
    print("   ✓ Knows when to be serious vs playful")
    print("=" * 60)
    print("🔥 Technical Features:")
    print("   ✓ Model: gemini-2.5-flash")
    print("   ✓ Enhanced system prompt with EQ")
    print("   ✓ Context-aware flavoring")
    print("   ✓ Sentiment-based responses")
    print("   ✓ Context memory (10 exchanges)")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)