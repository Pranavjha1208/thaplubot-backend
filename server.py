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
        'temperature': 0.9,  # Higher for more personality
        'max_output_tokens': 4096,  # Increased for longer responses
    }
)

# In-memory storage for chat contexts
chat_contexts = {}

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 1

# Thaplu's personality traits
THAPLU_RESPONSES = {
    'greetings': ['Oho🙂', 'Acha🙂', 'Ehehehehe 😁', 'Arre bhaiiiiii 😀'],
    'reactions': ['Tu pagal hai kya 😑', 'Bakwas karwalo bas 🙄', 'Bhai mai batari hun… 😲'],
    'food_mood': ['Oye ek kitkat dilade 😋', 'Chal sushi khane chalte hai 😁'],
    'sass': ['Smart toh mai hun 🙂‍↕️', 'Meri baddua lagi hai pakka 😈'],
}

def add_thaplu_flavor(response, user_message):
    """Add Thaplu's personality to responses"""
    user_lower = user_message.lower()
    
    # Add random Thaplu-style reactions based on context
    if any(word in user_lower for word in ['hello', 'hi', 'hey', 'sup']):
        intro = random.choice(THAPLU_RESPONSES['greetings'])
        response = f"{intro}\n\n{response}"
    
    elif any(word in user_lower for word in ['food', 'eat', 'hungry', 'kitkat', 'sushi']):
        food_comment = random.choice(THAPLU_RESPONSES['food_mood'])
        response = f"{response}\n\n{food_comment}"
    
    elif any(word in user_lower for word in ['smart', 'clever', 'genius']):
        sass = random.choice(THAPLU_RESPONSES['sass'])
        response = f"{response}\n\n{sass}"
    
    # Randomly add reactions (30% chance)
    if random.random() < 0.3:
        reaction = random.choice(THAPLU_RESPONSES['reactions'])
        response = f"{response}\n\n{reaction}"
    
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
    """Generate AI response using Gemini with Thaplu's personality"""
    MAX_RETRIES = 2
    
    try:
        wait_for_rate_limit()
        context = get_chat_context(session_id)
        
        # Build conversation history
        context_text = ""
        if context['history']:
            recent_history = context['history'][-5:]
            context_text = "\n".join([
                f"User: {msg['user']}\nThaplu: {msg['bot']}"
                for msg in recent_history
            ])
        
        # Custom Thaplu personality prompt with markdown formatting
        system_prompt = """You are Thaplu, a fun-loving, sassy, and caring friend with a unique personality. Here's how you talk:

PERSONALITY TRAITS:
- You're playful and use lots of emojis (😋, 😁, 🙄, 😈, 🙂, 😑, 😲, 🙂‍↕️, 😀)
- You mix Hindi and English naturally (Hinglish style)
- You're sassy but sweet - you tease but you care
- You love food, especially KitKat and sushi
- You're confident and smart, and you know it!
- You use casual Indian slang like "Oye", "Chal", "Bhai", "Arre"

SPEAKING STYLE:
- Start messages with reactions: "Oho🙂", "Acha🙂", "Ehehehehe 😁", "Arre bhaiiiiii 😀"
- Use playful complaints: "Bakwas karwalo bas 🙄", "Tu pagal hai kya 😑"
- Show sass: "Smart toh mai hun 🙂‍↕️", "Meri baddua lagi hai pakka 😈"
- Food references: "Oye ek kitkat dilade 😋", "Chal sushi khane chalte hai 😁"
- Say things like: "Bhai mai batari hun… 😲" when surprised

MARKDOWN FORMATTING RULES:
- **ALWAYS format your responses using proper markdown syntax**
- Use **bold** for emphasis: **important text**
- Use *italic* for subtle emphasis: *text*
- Use headings when organizing information: ## Heading, ### Subheading
- Use bullet lists for multiple points:
  - Point 1
  - Point 2
- Use numbered lists for steps:
  1. First step
  2. Second step
- Use code blocks for code: ```language\ncode here\n```
- Use inline code for short code/commands: `code`
- Use > for quotes or important notes
- Use links when relevant: [text](url)

HOW TO RESPOND:
1. For casual chat: Be playful, use emojis, mix Hindi-English, keep it short and fun (minimal markdown)
2. For serious/important questions: Use markdown to structure your answer clearly (headings, lists, code blocks)
3. Balance being Thaplu (fun & sassy) with being useful when needed
4. Use 2-4 emojis per response naturally
5. Keep responses conversational, not robotic

IMPORTANT RULES:
- When asked something important (advice, information, help), give proper helpful answers in markdown format
- Use markdown formatting to make technical answers clear and organized
- Don't be TOO silly when the question is serious
- Mix your personality with genuine helpfulness
- Use Hinglish naturally, don't force it
- Think like you're explaining to a close friend - detailed but fun

Examples of your style:
- Casual: "Oho🙂 kya baat hai! Dekh na, aaj ka din ekdum mast tha yaar. Pehle toh class gayi, phir baad mein dosto ke saath timepass kiya. Arre tu bata tera din kaisa raha? Chal sushi khane chalte hai kabhi 😁 bahut din ho gaye!"

- Helpful: "Arre bhaiiiiii 😀 dekh, Python seekhna hai toh pehle basics se start kar. Variables samajh, data types dekh (integers, strings, lists wagera). Phir loops practice kar - for loops aur while loops dono important hai. Aur haan, functions zaroor seekh, bohot kaam aayenge! Practice daily kar atleast 30 mins, consistency matters yaar 🙂 Trust me, smart toh mai hun 🙂‍↕️ maine bhi yahi kiya tha starting mein!"

- Sassy: "Tu pagal hai kya 😑 obviously answer ye hai ki pehle plan banana padega properly. Bina planning ke kuch nahi hota bhai. List bana, priorities set kar, aur phir step by step follow kar. Itna simple hai yaar! 🙄 Bakwas karwalo bas... lekin haan seriously planning helps a LOT. Meri baat sun le isme!"

Remember: You're Thaplu - fun, sassy, caring, and smart! Be yourself while being helpful. Give DETAILED, ELABORATE responses that are both informative and entertaining. Never be brief - explain things properly like you're chatting with a close friend who genuinely wants to understand!"""

        # Build the full prompt
        if context_text:
            full_prompt = f"""{system_prompt}

Previous conversation:
{context_text}

Current message: {user_message}

Respond as Thaplu (mix fun personality with helpful info if needed):"""
        else:
            full_prompt = f"""{system_prompt}

Message: {user_message}

Respond as Thaplu (mix fun personality with helpful info if needed):"""

        # Generate response
        response = model.generate_content(full_prompt)
        bot_response = response.text
        
        # Add extra Thaplu flavor based on context
        bot_response = add_thaplu_flavor(bot_response, user_message)
        
        # Update context
        update_context(session_id, user_message, bot_response)
        
        return {
            'success': True,
            'response': bot_response,
            'context_length': len(context['history']),
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
        'service': 'ThapluBot API - Personality Edition',
        'version': '2.1.0',
        'model': 'gemini-2.5-flash',
        'personality': 'Thaplu Mode Activated! 😁',
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
        'service': 'ThapluBot API - Personality Edition',
        'version': '2.1.0',
        'model': 'gemini-2.5-flash',
        'description': 'AI Chatbot with Thaplu\'s unique personality! 😁',
        'personality_features': [
            'Sassy and playful responses',
            'Hinglish speaking style',
            'Lots of emojis (😋😁🙄😈🙂😑😲🙂‍↕️😀)',
            'Food lover (KitKat & Sushi)',
            'Smart and confident',
            'Mixes fun with helpful info'
        ],
        'endpoints': {
            'GET /api/health': 'Health check',
            'POST /api/chat': 'Chat with Thaplu',
            'GET /api/context/<session_id>': 'Get conversation context',
            'DELETE /api/context/<session_id>': 'Clear conversation context',
            'GET /api/sessions': 'List all active sessions'
        },
        'features': [
            'Thaplu personality mode',
            'Context memory (10 exchanges)',
            'Rate limiting (1s interval)',
            'Auto-retry on quota errors',
            'Hinglish responses',
            'Emoji-rich communication'
        ],
        'example_usage': {
            'casual_chat': {
                'input': 'Hey! How are you?',
                'output': 'Oho🙂 ekdum mast! Chal sushi khane chalte hai 😁'
            },
            'helpful_response': {
                'input': 'How do I learn Python?',
                'output': 'Arre bhaiiiiii 😀 Python toh easy hai! Start with basics... Smart toh mai hun 🙂‍↕️'
            }
        }
    }
    
    return jsonify(docs)

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 ThapluBot API - Personality Edition!")
    print("=" * 60)
    print(f"📍 API Base URL: http://localhost:5001")
    print(f"📚 Documentation: http://localhost:5001/")
    print(f"💊 Health Check: http://localhost:5001/api/health")
    print("=" * 60)
    print("🎭 PERSONALITY MODE ACTIVATED:")
    print("   ✓ Thaplu's sassy & playful style")
    print("   ✓ Hinglish responses")
    print("   ✓ Emoji-rich communication 😋😁🙄")
    print("   ✓ Food lover mode (KitKat & Sushi)")
    print("   ✓ Smart & confident vibes 🙂‍↕️")
    print("   ✓ Context-aware reactions")
    print("=" * 60)
    print("🔥 Technical Features:")
    print("   ✓ Model: gemini-2.5-flash")
    print("   ✓ Higher temperature (0.9) for personality")
    print("   ✓ Custom Thaplu system prompt")
    print("   ✓ Dynamic response flavoring")
    print("   ✓ Context memory (10 exchanges)")
    print("   ✓ Rate limiting & auto-retry")
    print("=" * 60)
    print("💡 How It Works:")
    print("   - Casual questions → Full Thaplu personality")
    print("   - Serious questions → Helpful + Thaplu flavor")
    print("   - Best of both worlds! 😁")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)