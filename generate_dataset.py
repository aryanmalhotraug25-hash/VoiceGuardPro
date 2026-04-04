"""
=================================================================
 VoiceGuard Pro — Automated Dataset Generator
=================================================================
 Downloads real human speech + generates AI speech automatically.
 
 Install dependencies first:
   pip install edge-tts gtts pydub librosa soundfile datasets requests tqdm pyttsx3
   
 Then run:
   python generate_dataset.py
=================================================================
"""

import os
import sys
import asyncio
import random
import time
import warnings
warnings.filterwarnings('ignore')

# ── Directories ──
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REAL_DIR    = os.path.join(DATASET_DIR, "real")
FAKE_DIR    = os.path.join(DATASET_DIR, "fake")

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

print("=" * 65)
print("  VoiceGuard Pro — Dataset Generator")
print("=" * 65)


# ================================================================
# PART 1: DOWNLOAD REAL HUMAN VOICE SAMPLES
# ================================================================

# ── 100+ sentences for TTS generation ──
SENTENCES = [
    # Casual / everyday
    "Hey, good morning! How are you doing today?",
    "I just grabbed a coffee from the new cafe down the street.",
    "Can we reschedule our meeting to three thirty instead?",
    "I updated the presentation slides and sent them to your email.",
    "The weather looks great today, maybe we should go for a walk.",
    "Did you watch the match last night? It was incredible!",
    "I'm running a bit late, I'll be there in about ten minutes.",
    "Let me know when you're free, we can grab lunch together.",
    "I finished reading that book you recommended, it was amazing.",
    "Happy birthday! I hope you have a wonderful day.",
    "Could you please send me the report by end of day?",
    "I think we should take a different approach to this problem.",
    "The kids are doing great in school this semester.",
    "I'll pick up some groceries on my way home tonight.",
    "That new restaurant downtown has really good food.",
    "We should plan a trip sometime this summer.",
    "I've been working on this project for about three weeks now.",
    "The traffic was terrible this morning, it took me an hour.",
    "Can you help me move this weekend? I'll buy pizza.",
    "I just saw the funniest video, let me send it to you.",
    "My phone battery is about to die, I'll call you back later.",
    "The meeting went really well, the client loved our proposal.",
    "I need to pick up my prescription from the pharmacy.",
    "Have you tried that new app everyone's been talking about?",
    "I'm thinking of learning to play the guitar this year.",
    "The garden is looking beautiful with all the flowers blooming.",
    "We should catch up soon, it's been way too long.",
    "I'm making pasta for dinner, would you like to join us?",
    "The new season of that show starts next week.",
    "I finally fixed the leaky faucet in the bathroom.",
    
    # Professional / work
    "Good afternoon, I'm calling regarding the project update.",
    "The quarterly results show a fifteen percent increase in revenue.",
    "We need to finalize the budget before the board meeting.",
    "I've scheduled a code review for tomorrow at two o'clock.",
    "The deployment went smoothly, no issues reported so far.",
    "Please review the pull request when you get a chance.",
    "Our team completed the sprint ahead of schedule.",
    "The client has approved the final design mockups.",
    "We should consider implementing automated testing.",
    "The server migration is planned for this weekend.",
    "I'll prepare the documentation for the new API endpoints.",
    "The performance optimization reduced load time by forty percent.",
    "We received positive feedback from the user testing sessions.",
    "The database backup completed successfully last night.",
    "I recommend we upgrade our infrastructure before the launch.",
    
    # Informational / neutral
    "The temperature today is expected to reach thirty two degrees.",
    "The train to the city center departs every fifteen minutes.",
    "The library closes at eight o'clock on weekdays.",
    "The nearest hospital is about five kilometers from here.",
    "The population of the city has grown significantly.",
    "Solar panels can reduce electricity bills substantially.",
    "Regular exercise helps improve cardiovascular health.",
    "The speed limit on this highway is one hundred kilometers.",
    "Recycling helps reduce waste and protect the environment.",
    "The museum has a new exhibition opening this Friday.",
    
    # Emotional / expressive
    "I'm so excited about the concert this weekend!",
    "That was absolutely the best movie I've ever seen.",
    "I can't believe we actually won the championship!",
    "I'm really sorry to hear about what happened.",
    "This is so frustrating, nothing seems to be working.",
    "Thank you so much for helping me with everything.",
    "I'm a little nervous about the presentation tomorrow.",
    "Wow, this view is absolutely breathtaking!",
    "I'm really proud of what we've accomplished together.",
    "It breaks my heart to see them go through that.",
    
    # Questions and dialogue
    "What time does the store close on Sundays?",
    "Have you finished the assignment that's due tomorrow?",
    "Where would you like to go for dinner tonight?",
    "Do you know if the package has been delivered yet?",
    "How long have you been working at your current job?",
    "What do you think about the new company policy?",
    "Can you explain how this feature works?",
    "Would you prefer tea or coffee?",
    "Is there anything else I can help you with?",
    "When was the last time you went on vacation?",
    
    # Longer / complex sentences
    "I was thinking about what you said yesterday, and I completely agree that we should focus more on quality rather than quantity.",
    "The research paper discusses how artificial intelligence is transforming healthcare, particularly in the areas of diagnosis and treatment planning.",
    "After careful consideration of all the options, the committee decided to move forward with the renovation project starting next quarter.",
    "If you could travel anywhere in the world without worrying about cost or time, where would you choose to go and why?",
    "The documentary explores the impact of climate change on coastal communities and offers practical solutions for adaptation.",
    "During our vacation last summer, we visited several historical sites and learned so much about the local culture and traditions.",
    "The new software update includes several bug fixes, performance improvements, and a completely redesigned user interface.",
    "I remember when we used to play in the park as children, those were some of the happiest days of my life.",
    "The professor explained that understanding basic statistical concepts is essential for making informed decisions in everyday life.",
    "Looking back at all we've accomplished this year, I think we should be really proud of the team's dedication and hard work.",
    
    # Short phrases (for variety in length)
    "Hello, how are you?",
    "That sounds great.",
    "I'll be right there.",
    "No problem at all.",
    "See you tomorrow.",
    "Good night, take care.",
    "Let me think about it.",
    "Absolutely, count me in.",
    "I completely understand.",
    "That makes total sense.",
]

# ── Scam-related sentences (for fake voice training) ──
SCAM_SENTENCES = [
    "This is an urgent call regarding your bank account security.",
    "Your account has been compromised and we need immediate verification.",
    "Please share your OTP to complete the verification process.",
    "Transfer the amount immediately to avoid legal action.",
    "You have won a lottery prize of fifty thousand rupees.",
    "Your son has been in a serious accident at the hospital.",
    "This is the police department, there is a warrant in your name.",
    "Your Aadhaar card has been used for illegal activities.",
    "Press one to speak with our fraud prevention department.",
    "Your credit card will be blocked if you don't verify now.",
    "We are calling from the income tax department about your case.",
    "You must pay the fine within one hour or face arrest.",
    "Your KYC has expired, update it now or your account will be frozen.",
    "We detected suspicious activity on your UPI account.",
    "Please provide your ATM pin for account verification.",
    "This is your last warning before we take legal action.",
    "A complaint has been filed against your PAN card number.",
    "Your insurance policy is about to expire, renew it immediately.",
    "We are from the RBI and your account is under investigation.",
    "Share your bank details to receive your refund of ten thousand rupees.",
    "Someone tried to withdraw money from your account, verify now.",
    "You have an outstanding loan payment, pay immediately.",
    "Your mobile number will be disconnected in twenty four hours.",
    "This call is being recorded for legal and compliance purposes.",
    "Failure to comply will result in immediate suspension of services.",
]


def download_real_voices_librispeech():
    """Download real human voice samples from LibriSpeech via HuggingFace."""
    print("\n  📥 Downloading REAL human voices from LibriSpeech...")
    
    try:
        from datasets import load_dataset
        import soundfile as sf
        
        # Load LibriSpeech clean subset
        print("     Loading dataset (this may take a few minutes on first run)...")
        dataset = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="train.100",
            trust_remote_code=True,
            streaming=True  # Stream to avoid downloading entire dataset
        )
        
        count = 0
        target = 150  # Number of real samples to download
        
        for sample in dataset:
            if count >= target:
                break
            
            try:
                audio = sample["audio"]
                y = audio["array"]
                sr = audio["sampling_rate"]
                
                # Skip very short clips (< 2 seconds)
                if len(y) / sr < 2.0:
                    continue
                
                # Skip very long clips (> 15 seconds) — trim them
                max_samples = int(15.0 * sr)
                if len(y) > max_samples:
                    y = y[:max_samples]
                
                # Normalize
                y = y / (max(abs(y.max()), abs(y.min())) + 1e-8)
                y = y.astype('float32')
                
                outpath = os.path.join(REAL_DIR, f"real_libri_{count:04d}.wav")
                sf.write(outpath, y, sr)
                count += 1
                
                if count % 25 == 0:
                    print(f"     ✅ Downloaded {count}/{target} real samples")
                    
            except Exception as e:
                continue
        
        print(f"     ✅ Downloaded {count} real human voice samples")
        return count
        
    except ImportError:
        print("     ⚠️ 'datasets' not installed. Run: pip install datasets")
        return 0
    except Exception as e:
        print(f"     ❌ LibriSpeech download failed: {e}")
        return 0


def download_real_voices_lj():
    """Alternative: Download from LJ Speech dataset."""
    print("\n  📥 Trying LJ Speech dataset as backup...")
    
    try:
        from datasets import load_dataset
        import soundfile as sf
        
        dataset = load_dataset(
            "lj_speech",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        
        count = 0
        target = 100
        
        for sample in dataset:
            if count >= target:
                break
            try:
                audio = sample["audio"]
                y = audio["array"]
                sr = audio["sampling_rate"]
                
                if len(y) / sr < 2.0:
                    continue
                
                max_samples = int(15.0 * sr)
                if len(y) > max_samples:
                    y = y[:max_samples]
                
                y = y / (max(abs(y.max()), abs(y.min())) + 1e-8)
                y = y.astype('float32')
                
                outpath = os.path.join(REAL_DIR, f"real_lj_{count:04d}.wav")
                sf.write(outpath, y, sr)
                count += 1
                
                if count % 25 == 0:
                    print(f"     ✅ Downloaded {count}/{target} LJ samples")
            except:
                continue
        
        print(f"     ✅ Downloaded {count} LJ Speech samples")
        return count
        
    except Exception as e:
        print(f"     ❌ LJ Speech failed: {e}")
        return 0


def download_real_voices_common_voice():
    """Alternative: Download from Mozilla Common Voice."""
    print("\n  📥 Trying Common Voice dataset...")
    
    try:
        from datasets import load_dataset
        import soundfile as sf
        
        dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True,
            token=None,  # May need HuggingFace token
        )
        
        count = 0
        target = 100
        
        for sample in dataset:
            if count >= target:
                break
            try:
                audio = sample["audio"]
                y = audio["array"]
                sr = audio["sampling_rate"]
                
                if len(y) / sr < 2.0:
                    continue
                
                max_samples = int(15.0 * sr)
                if len(y) > max_samples:
                    y = y[:max_samples]
                
                y = y / (max(abs(y.max()), abs(y.min())) + 1e-8)
                y = y.astype('float32')
                
                outpath = os.path.join(REAL_DIR, f"real_cv_{count:04d}.wav")
                sf.write(outpath, y, sr)
                count += 1
                
                if count % 25 == 0:
                    print(f"     ✅ Downloaded {count}/{target} Common Voice samples")
            except:
                continue
        
        print(f"     ✅ Downloaded {count} Common Voice samples")
        return count
        
    except Exception as e:
        print(f"     ❌ Common Voice failed: {e}")
        return 0


# ================================================================
# PART 2: GENERATE AI/FAKE VOICE SAMPLES
# ================================================================

async def generate_edge_tts_samples():
    """Generate AI voice samples using Microsoft Edge TTS (free, high quality)."""
    print("\n  🤖 Generating FAKE AI voices using Edge-TTS...")
    
    try:
        import edge_tts
    except ImportError:
        print("     ⚠️ edge-tts not installed. Run: pip install edge-tts")
        return 0
    
    # Diverse set of neural voices (these sound very realistic)
    VOICES = [
        # English (US)
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-AriaNeural",
        "en-US-DavisNeural",
        "en-US-AmberNeural",
        "en-US-AnaNeural",
        "en-US-AndrewNeural",
        "en-US-BrandonNeural",
        "en-US-ChristopherNeural",
        "en-US-CoraNeural",
        "en-US-ElizabethNeural",
        "en-US-EricNeural",
        "en-US-JacobNeural",
        "en-US-JaneNeural",
        "en-US-JasonNeural",
        "en-US-MichelleNeural",
        "en-US-MonicaNeural",
        "en-US-NancyNeural",
        "en-US-RogerNeural",
        "en-US-SaraNeural",
        "en-US-SteffanNeural",
        "en-US-TonyNeural",
        
        # English (India)
        "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural",
        "en-IN-NeerjaExpressiveNeural",
        
        # English (UK)
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-GB-LibbyNeural",
        "en-GB-AbbiNeural",
        "en-GB-AlfieNeural",
        "en-GB-BellaNeural",
        "en-GB-ElliotNeural",
        "en-GB-EthanNeural",
        "en-GB-HollieNeural",
        "en-GB-MaisieNeural",
        "en-GB-NoahNeural",
        "en-GB-OliverNeural",
        "en-GB-OliviaNeural",
        "en-GB-ThomasNeural",
        
        # English (Australia)
        "en-AU-NatashaNeural",
        "en-AU-WilliamNeural",
        "en-AU-AnnetteNeural",
        "en-AU-CarlyNeural",
        "en-AU-DarrenNeural",
        "en-AU-DuncanNeural",
        "en-AU-ElsieNeural",
        "en-AU-FreyaNeural",
        "en-AU-JoanneNeural",
        "en-AU-KenNeural",
        "en-AU-KimNeural",
        "en-AU-NeilNeural",
        "en-AU-TimNeural",
        "en-AU-TinaNeural",
    ]
    
    # Combine all sentences
    all_sentences = SENTENCES + SCAM_SENTENCES
    
    count = 0
    total_target = min(len(all_sentences) * 3, 300)  # Generate up to 300 samples
    
    for i, text in enumerate(all_sentences):
        # Use 2-3 different voices per sentence
        selected_voices = random.sample(VOICES, min(3, len(VOICES)))
        
        for voice in selected_voices:
            if count >= total_target:
                break
            
            outpath = os.path.join(FAKE_DIR, f"fake_edge_{count:04d}.wav")
            
            try:
                # Generate with different speech rates for variety
                rate = random.choice(["-10%", "-5%", "+0%", "+5%", "+10%"])
                pitch = random.choice(["-5Hz", "+0Hz", "+5Hz"])
                
                communicate = edge_tts.Communicate(
                    text, voice, rate=rate, pitch=pitch
                )
                
                # Save as mp3 first, then we'll convert
                mp3_path = outpath.replace('.wav', '.mp3')
                await communicate.save(mp3_path)
                
                # Convert mp3 to wav
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(mp3_path)
                    audio = audio.set_channels(1).set_frame_rate(22050)
                    audio.export(outpath, format="wav")
                    os.remove(mp3_path)
                except:
                    # If pydub fails, keep the mp3 with wav extension
                    # (librosa can load mp3 too)
                    os.rename(mp3_path, outpath)
                
                count += 1
                
                if count % 25 == 0:
                    print(f"     ✅ Generated {count}/{total_target} AI voice samples")
                    
            except Exception as e:
                # Skip errors silently
                if os.path.exists(outpath):
                    try: os.remove(outpath)
                    except: pass
                mp3_path = outpath.replace('.wav', '.mp3')
                if os.path.exists(mp3_path):
                    try: os.remove(mp3_path)
                    except: pass
                continue
        
        if count >= total_target:
            break
    
    print(f"     ✅ Generated {count} Edge-TTS AI voice samples")
    return count


def generate_gtts_samples():
    """Generate AI voice samples using Google TTS (free, different quality)."""
    print("\n  🤖 Generating FAKE AI voices using Google TTS...")
    
    try:
        from gtts import gTTS
    except ImportError:
        print("     ⚠️ gtts not installed. Run: pip install gtts")
        return 0
    
    # Languages/accents available in gTTS
    ACCENTS = [
        ("en", "com"),      # US English
        ("en", "co.uk"),    # UK English
        ("en", "co.in"),    # Indian English
        ("en", "com.au"),   # Australian English
        ("en", "co.za"),    # South African English
    ]
    
    all_sentences = SENTENCES[:40] + SCAM_SENTENCES[:15]  # Use subset
    count = 0
    
    for i, text in enumerate(all_sentences):
        for lang, tld in ACCENTS:
            outpath = os.path.join(FAKE_DIR, f"fake_gtts_{count:04d}.wav")
            
            try:
                tts = gTTS(text=text, lang=lang, tld=tld, slow=random.choice([True, False]))
                mp3_path = outpath.replace('.wav', '.mp3')
                tts.save(mp3_path)
                
                # Convert to wav
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(mp3_path)
                    audio = audio.set_channels(1).set_frame_rate(22050)
                    audio.export(outpath, format="wav")
                    os.remove(mp3_path)
                except:
                    os.rename(mp3_path, outpath)
                
                count += 1
                
                if count % 25 == 0:
                    print(f"     ✅ Generated {count} gTTS samples")
                    
            except Exception:
                continue
            
            # Small delay to avoid rate limiting
            time.sleep(0.3)
    
    print(f"     ✅ Generated {count} Google TTS AI voice samples")
    return count


def generate_pyttsx3_samples():
    """Generate AI voice samples using pyttsx3 (offline, robotic)."""
    print("\n  🤖 Generating FAKE AI voices using pyttsx3 (offline)...")
    
    try:
        import pyttsx3
    except ImportError:
        print("     ⚠️ pyttsx3 not installed. Run: pip install pyttsx3")
        return 0
    
    try:
        import soundfile as sf
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        if not voices:
            print("     ⚠️ No voices found for pyttsx3")
            return 0
        
        count = 0
        sentences_subset = SENTENCES[:30] + SCAM_SENTENCES[:10]
        
        for i, text in enumerate(sentences_subset):
            for vi, voice in enumerate(voices[:4]):  # Use up to 4 system voices
                for rate in [130, 160, 190, 220]:  # Different speeds
                    outpath = os.path.join(FAKE_DIR, f"fake_pyttsx_{count:04d}.wav")
                    
                    try:
                        engine.setProperty('voice', voice.id)
                        engine.setProperty('rate', rate)
                        engine.save_to_file(text, outpath)
                        engine.runAndWait()
                        
                        if os.path.isfile(outpath) and os.path.getsize(outpath) > 1000:
                            count += 1
                    except:
                        continue
                    
                    if count % 25 == 0 and count > 0:
                        print(f"     ✅ Generated {count} pyttsx3 samples")
        
        print(f"     ✅ Generated {count} pyttsx3 AI voice samples")
        return count
        
    except Exception as e:
        print(f"     ❌ pyttsx3 generation failed: {e}")
        return 0


# ================================================================
# PART 3: GENERATE ADDITIONAL REAL SAMPLES BY RECORDING TIPS
# ================================================================

def create_recording_guide():
    """Create a guide file for manual recording."""
    guide_path = os.path.join(DATASET_DIR, "RECORDING_GUIDE.txt")
    
    guide = """
╔══════════════════════════════════════════════════════════════╗
║          VoiceGuard — Manual Recording Guide                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  For BEST accuracy, also record real human voices manually:  ║
║                                                              ║
║  1. Use your phone's voice recorder app                      ║
║  2. Record 50+ clips of 5-15 seconds each                   ║
║  3. Get recordings from DIFFERENT people:                    ║
║     - Male and female voices                                 ║
║     - Different ages (young, middle-aged, elderly)           ║
║     - Different accents                                      ║
║     - Quiet room + noisy environments                        ║
║  4. Save as .wav files in the dataset/real/ folder           ║
║                                                              ║
║  Recording tips:                                             ║
║  - Speak naturally (don't read robotically)                  ║
║  - Include pauses, "um"s, and natural hesitations            ║
║  - Vary your volume and emotion                              ║
║  - Include some phone-quality recordings                     ║
║                                                              ║
║  Say things like:                                            ║
║  - "Hey, how's it going? I was just calling to check in."   ║
║  - "So um, about the meeting tomorrow, can we push it?"     ║
║  - "Oh my god, did you hear what happened? It's crazy!"     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    with open(guide_path, 'w') as f:
        f.write(guide)
    print(f"  📄 Recording guide saved to: {guide_path}")


# ================================================================
# PART 4: VALIDATION
# ================================================================

def validate_dataset():
    """Check the generated dataset."""
    print("\n" + "=" * 65)
    print("  📊 Dataset Validation")
    print("=" * 65)
    
    real_files = [f for f in os.listdir(REAL_DIR) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    fake_files = [f for f in os.listdir(FAKE_DIR) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    
    print(f"\n  Real (human) samples: {len(real_files)}")
    print(f"  Fake (AI) samples:    {len(fake_files)}")
    print(f"  Total:                {len(real_files) + len(fake_files)}")
    
    # Check sizes
    real_sizes = []
    for f in real_files[:10]:
        size = os.path.getsize(os.path.join(REAL_DIR, f))
        real_sizes.append(size)
    
    fake_sizes = []
    for f in fake_files[:10]:
        size = os.path.getsize(os.path.join(FAKE_DIR, f))
        fake_sizes.append(size)
    
    if real_sizes:
        avg_real = sum(real_sizes) / len(real_sizes) / 1024
        print(f"\n  Avg real file size: {avg_real:.1f} KB")
    if fake_sizes:
        avg_fake = sum(fake_sizes) / len(fake_sizes) / 1024
        print(f"  Avg fake file size: {avg_fake:.1f} KB")
    
    # Verify audio can be loaded
    print("\n  Verifying audio files can be loaded...")
    try:
        import librosa
        
        errors = 0
        checked = 0
        for folder, files in [(REAL_DIR, real_files[:5]), (FAKE_DIR, fake_files[:5])]:
            for f in files:
                try:
                    y, sr = librosa.load(os.path.join(folder, f), sr=22050, duration=5)
                    if len(y) < 22050 * 0.5:
                        print(f"     ⚠️ Too short: {f}")
                        errors += 1
                    checked += 1
                except Exception as e:
                    print(f"     ❌ Cannot load: {f} — {e}")
                    errors += 1
        
        if errors == 0:
            print(f"     ✅ All {checked} checked files load correctly")
        else:
            print(f"     ⚠️ {errors}/{checked} files had issues")
            
    except ImportError:
        print("     ⚠️ librosa not installed, skipping verification")
    
    # Recommendations
    print("\n  📋 Recommendations:")
    
    if len(real_files) < 50:
        print(f"     ⚠️ Only {len(real_files)} real samples — aim for 100+")
        print(f"        → Record yourself & friends speaking naturally")
        print(f"        → Download more from LibriSpeech or Common Voice")
    else:
        print(f"     ✅ Good number of real samples")
    
    if len(fake_files) < 50:
        print(f"     ⚠️ Only {len(fake_files)} fake samples — aim for 100+")
        print(f"        → Run Edge-TTS generation again with more sentences")
    else:
        print(f"     ✅ Good number of fake samples")
    
    ratio = len(real_files) / (len(fake_files) + 1)
    if ratio < 0.5 or ratio > 2.0:
        print(f"     ⚠️ Imbalanced dataset (ratio: {ratio:.1f}). Try to balance 1:1")
    else:
        print(f"     ✅ Dataset is reasonably balanced (ratio: {ratio:.1f})")
    
    if len(real_files) >= 50 and len(fake_files) >= 50:
        print(f"\n  🎉 Dataset is ready! Run: python train_model.py")
    
    return len(real_files), len(fake_files)


# ================================================================
# MAIN EXECUTION
# ================================================================

async def main():
    total_real = 0
    total_fake = 0
    
    # ── Step 1: Download real human voices ──
    print("\n" + "─" * 65)
    print("  STEP 1: Getting REAL human voice samples")
    print("─" * 65)
    
    # Try LibriSpeech first
    n = download_real_voices_librispeech()
    total_real += n
    
    # If not enough, try LJ Speech
    if total_real < 80:
        n = download_real_voices_lj()
        total_real += n
    
    # If still not enough, try Common Voice
    if total_real < 50:
        n = download_real_voices_common_voice()
        total_real += n
    
    if total_real == 0:
        print("\n  ⚠️ Could not download real voice samples automatically.")
        print("     Please manually add .wav files to: dataset/real/")
        print("     Sources:")
        print("       • https://commonvoice.mozilla.org (free download)")
        print("       • https://www.openslr.org/12/ (LibriSpeech)")
        print("       • Record yourself and friends speaking!")
    
    # ── Step 2: Generate AI voices ──
    print("\n" + "─" * 65)
    print("  STEP 2: Generating FAKE AI voice samples")
    print("─" * 65)
    
    # Edge TTS (best quality, free)
    n = await generate_edge_tts_samples()
    total_fake += n
    
    # Google TTS (different quality profile)
    if total_fake < 100:
        n = generate_gtts_samples()
        total_fake += n
    
    # pyttsx3 (offline, robotic — adds diversity)
    if total_fake < 150:
        n = generate_pyttsx3_samples()
        total_fake += n
    
    # ── Step 3: Create recording guide ──
    create_recording_guide()
    
    # ── Step 4: Validate ──
    n_real, n_fake = validate_dataset()
    
    # ── Summary ──
    print("\n" + "=" * 65)
    print("  ✅ DATASET GENERATION COMPLETE")
    print("=" * 65)
    print(f"  Real samples: {n_real}")
    print(f"  Fake samples: {n_fake}")
    print(f"  Total:        {n_real + n_fake}")
    print(f"\n  Next steps:")
    print(f"    1. python train_model.py    ← Train the model")
    print(f"    2. streamlit run app.py     ← Launch the app")
    print("=" * 65)


if __name__ == "__main__":
    # Handle async for edge-tts
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())