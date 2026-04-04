"""
=================================================================
 VoiceGuard Pro — Real Human Voice Downloader (ROBUST)
=================================================================
 5 methods with automatic fallbacks. At least one WILL work.
 
 Install:
   pip install datasets soundfile librosa requests tqdm
   
 Run:
   python download_real_voices.py
=================================================================
"""

import os, sys, time, tempfile, struct, wave
import numpy as np

REAL_DIR = os.path.join("dataset", "real")
os.makedirs(REAL_DIR, exist_ok=True)

TARGET = 150
EXTENSIONS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')


def count_existing():
    if not os.path.isdir(REAL_DIR):
        return 0
    return len([f for f in os.listdir(REAL_DIR) if f.lower().endswith(EXTENSIONS)])


def save_audio_wav(y, sr, filename):
    """Save numpy array as WAV — works even without soundfile."""
    y = np.array(y, dtype=np.float32)
    peak = max(abs(y.max()), abs(y.min()), 1e-8)
    y = y / peak * 0.92  # normalize
    filepath = os.path.join(REAL_DIR, filename)

    # Try soundfile first
    try:
        import soundfile as sf
        sf.write(filepath, y, sr)
        return True
    except:
        pass

    # Fallback: manual WAV writer
    try:
        y_int16 = (y * 32767).astype(np.int16)
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(y_int16.tobytes())
        return True
    except:
        return False


# ================================================================
# METHOD 1: Google FLEURS (most accessible — NO auth needed)
# ================================================================
def method_1_fleurs(needed):
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  Method 1: Google FLEURS Dataset                  ║")
    print("  ╚═══════════════════════════════════════════════════╝")

    try:
        from datasets import load_dataset
    except ImportError:
        print("    ❌ 'datasets' not installed → pip install datasets")
        return 0

    try:
        print("    ⏳ Connecting to HuggingFace (streaming mode)...")
        ds = load_dataset(
            "google/fleurs",
            "en_us",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        errors = 0
        for sample in ds:
            if count >= needed:
                break
            if errors > 50:
                print("    ⚠️ Too many errors, moving to next method")
                break

            try:
                audio = sample["audio"]
                y = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]

                duration = len(y) / sr
                if duration < 2.0:
                    continue
                if duration > 15.0:
                    y = y[:int(15.0 * sr)]

                if save_audio_wav(y, sr, f"real_fleurs_{count:04d}.wav"):
                    count += 1
                    if count % 20 == 0:
                        print(f"    ✅ Downloaded {count}/{needed} samples")
            except Exception:
                errors += 1
                continue

        print(f"    ✅ Got {count} samples from FLEURS")
        return count

    except Exception as e:
        print(f"    ❌ Failed: {str(e)[:100]}")
        return 0


# ================================================================
# METHOD 2: LibriSpeech (most popular speech dataset)
# ================================================================
def method_2_librispeech(needed):
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  Method 2: LibriSpeech ASR Dataset                ║")
    print("  ╚═══════════════════════════════════════════════════╝")

    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    # Try multiple split names (API changes between versions)
    splits_to_try = [
        ("librispeech_asr", "clean", "test"),
        ("librispeech_asr", "clean", "validation"),
        ("librispeech_asr", "other", "test"),
    ]

    for ds_name, config, split in splits_to_try:
        try:
            print(f"    ⏳ Trying {ds_name}/{config}/{split}...")
            ds = load_dataset(
                ds_name,
                config,
                split=split,
                streaming=True,
                trust_remote_code=True,
            )

            count = 0
            errors = 0
            for sample in ds:
                if count >= needed:
                    break
                if errors > 30:
                    break

                try:
                    audio = sample["audio"]
                    y = np.array(audio["array"], dtype=np.float32)
                    sr = audio["sampling_rate"]

                    duration = len(y) / sr
                    if duration < 2.0:
                        continue
                    if duration > 15.0:
                        y = y[:int(15.0 * sr)]

                    if save_audio_wav(y, sr, f"real_libri_{count:04d}.wav"):
                        count += 1
                        if count % 20 == 0:
                            print(f"    ✅ Downloaded {count}/{needed} samples")
                except:
                    errors += 1
                    continue

            if count > 0:
                print(f"    ✅ Got {count} samples from LibriSpeech")
                return count

        except Exception as e:
            print(f"    ⚠️ {ds_name}/{config}/{split} failed: {str(e)[:80]}")
            continue

    print("    ❌ All LibriSpeech attempts failed")
    return 0


# ================================================================
# METHOD 3: VoxPopuli (European Parliament — diverse speakers)
# ================================================================
def method_3_voxpopuli(needed):
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  Method 3: Facebook VoxPopuli Dataset              ║")
    print("  ╚═══════════════════════════════════════════════════╝")

    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    try:
        print("    ⏳ Connecting...")
        ds = load_dataset(
            "facebook/voxpopuli",
            "en",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        errors = 0
        for sample in ds:
            if count >= needed:
                break
            if errors > 30:
                break

            try:
                audio = sample["audio"]
                y = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]

                duration = len(y) / sr
                if duration < 2.0:
                    continue
                if duration > 15.0:
                    y = y[:int(15.0 * sr)]

                if save_audio_wav(y, sr, f"real_voxpop_{count:04d}.wav"):
                    count += 1
                    if count % 20 == 0:
                        print(f"    ✅ Downloaded {count}/{needed} samples")
            except:
                errors += 1
                continue

        print(f"    ✅ Got {count} from VoxPopuli")
        return count

    except Exception as e:
        print(f"    ❌ Failed: {str(e)[:100]}")
        return 0


# ================================================================
# METHOD 4: Direct HTTP Download from OpenSLR (MOST RELIABLE)
# ================================================================
def method_4_direct_download(needed):
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  Method 4: Direct Download from OpenSLR            ║")
    print("  ║  (Most reliable — plain HTTP, no API needed)       ║")
    print("  ╚═══════════════════════════════════════════════════╝")

    import urllib.request
    import tarfile
    import io

    # LibriSpeech dev-clean is ~338MB — contains 2703 utterances
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    tar_path = os.path.join("dataset", "dev-clean.tar.gz")

    try:
        # Check if already downloaded
        if os.path.isfile(tar_path) and os.path.getsize(tar_path) > 100_000_000:
            print("    📦 Found existing download, extracting...")
        else:
            print(f"    ⏳ Downloading LibriSpeech dev-clean (~338 MB)...")
            print(f"    📡 URL: {url}")
            print(f"    ⏱️  This may take 3-10 minutes depending on connection...")

            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(downloaded / total_size * 100, 100)
                    mb = downloaded / 1024 / 1024
                    total_mb = total_size / 1024 / 1024
                    print(f"\r    📥 {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)

            urllib.request.urlretrieve(url, tar_path, reporthook=report_progress)
            print()  # newline after progress

        # Extract FLAC files
        print("    📦 Extracting audio files...")
        count = 0

        try:
            import librosa
            USE_LIBROSA = True
        except ImportError:
            USE_LIBROSA = False

        with tarfile.open(tar_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".flac")]
            print(f"    📁 Found {len(members)} audio files in archive")

            for member in members:
                if count >= needed:
                    break

                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue

                    # Save FLAC temporarily, then convert
                    tmp_path = os.path.join(REAL_DIR, "temp_extract.flac")
                    with open(tmp_path, 'wb') as tmp:
                        tmp.write(f.read())

                    # Convert FLAC to WAV
                    if USE_LIBROSA:
                        y, sr = librosa.load(tmp_path, sr=22050, duration=15)
                    else:
                        import soundfile as sf
                        y, sr = sf.read(tmp_path)
                        y = y.astype(np.float32)

                    os.remove(tmp_path)

                    duration = len(y) / sr
                    if duration < 2.0:
                        continue

                    out_name = f"real_openslr_{count:04d}.wav"
                    if save_audio_wav(y, 22050 if USE_LIBROSA else sr, out_name):
                        count += 1
                        if count % 25 == 0:
                            print(f"    ✅ Extracted {count}/{needed} samples")

                except Exception:
                    continue

        # Clean up tar file to save space
        try:
            if count > 50:
                os.remove(tar_path)
                print(f"    🗑️  Cleaned up archive to save disk space")
        except:
            pass

        print(f"    ✅ Got {count} samples from OpenSLR direct download")
        return count

    except Exception as e:
        print(f"    ❌ Failed: {str(e)[:120]}")
        # Clean up partial download
        if os.path.isfile(tar_path) and os.path.getsize(tar_path) < 100_000_000:
            try:
                os.remove(tar_path)
            except:
                pass
        return 0


# ================================================================
# METHOD 6: Generate from YouTube Public Domain (yt-dlp)
# ================================================================
def method_6_youtube_speeches(needed):
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  Method 6: Public Domain Speeches (yt-dlp)         ║")
    print("  ╚═══════════════════════════════════════════════════╝")

    try:
        import subprocess
        result = subprocess.run(["yt-dlp", "--version"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("    ❌ yt-dlp not installed → pip install yt-dlp")
        return 0

    # Public domain speech videos (lectures, speeches, TED-style)
    # These are Creative Commons / Public Domain
    PUBLIC_VIDEOS = [
        # Public domain speeches and lectures
        "https://www.youtube.com/watch?v=zGbKOSgglG4",  # Public domain speech
        "https://www.youtube.com/watch?v=UF8uR6Z6KLc",  # Steve Jobs Stanford (widely shared)
        "https://www.youtube.com/watch?v=_MBgz9h7GGM",  # TED talk
    ]

    count = 0
    try:
        import librosa
    except ImportError:
        print("    ❌ librosa needed for this method")
        return 0

    for vid_url in PUBLIC_VIDEOS:
        if count >= needed:
            break

        try:
            tmp_audio = os.path.join("dataset", f"_yt_temp_{count}.wav")

            # Download audio only
            subprocess.run([
                "yt-dlp",
                "-x", "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", tmp_audio.replace('.wav', '.%(ext)s'),
                "--max-downloads", "1",
                "--no-playlist",
                vid_url
            ], capture_output=True, timeout=120)

            # Find the downloaded file
            possible = [f for f in os.listdir("dataset") if f.startswith("_yt_temp_")]
            if not possible:
                continue

            dl_path = os.path.join("dataset", possible[0])
            y, sr = librosa.load(dl_path, sr=22050, duration=300)

            # Split into 5-10 second segments
            seg_len = int(7.0 * sr)
            hop = int(5.0 * sr)

            for start in range(0, len(y) - seg_len, hop):
                if count >= needed:
                    break

                segment = y[start:start + seg_len]

                # Skip silence
                if np.mean(np.abs(segment)) < 0.01:
                    continue

                if save_audio_wav(segment, sr, f"real_speech_{count:04d}.wav"):
                    count += 1

            # Clean up
            os.remove(dl_path)

            if count % 10 == 0 and count > 0:
                print(f"    ✅ Extracted {count} segments")

        except Exception:
            continue

    print(f"    ✅ Got {count} from YouTube speeches")
    return count


# ================================================================
# METHOD 7: Record from Microphone (GUARANTEED to work)
# ================================================================
def method_7_microphone(needed):
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  Method 7: Record from Microphone (Interactive)    ║")
    print("  ╚═══════════════════════════════════════════════════╝")

    try:
        import sounddevice as sd
    except ImportError:
        print("    ❌ sounddevice not installed → pip install sounddevice")
        return 0

    sr = 22050
    duration = 5  # seconds per clip
    count = 0

    print(f"\n    🎤 We'll record {needed} clips of {duration}s each.")
    print(f"    📢 Speak NATURALLY — include pauses, 'um's, varied tone.")
    print(f"    ⏎  Press ENTER to start each recording, or 'q' to quit.\n")

    prompts = [
        "Say: 'Hey, good morning! How's it going?'",
        "Say: 'I was thinking we could grab lunch later.'",
        "Say: 'The meeting went really well yesterday.'",
        "Say: 'Can you send me that file when you get a chance?'",
        "Say: 'Oh wow, I didn't expect that at all!'",
        "Say: 'Let me check my calendar and get back to you.'",
        "Say anything naturally — tell a short story.",
        "Say anything — describe what you see around you.",
        "Say: 'I'm running a bit late, be there in ten.'",
        "Say anything — talk about your favorite food.",
        "Say: 'That's a great idea, let's do it.'",
        "Say something in a happy/excited tone.",
        "Say something in a calm/tired tone.",
        "Say anything — describe your day so far.",
        "Say: 'Sorry I missed your call earlier.'",
    ]

    while count < needed:
        prompt = prompts[count % len(prompts)]
        print(f"    [{count+1}/{needed}] {prompt}")
        user_input = input("    Press ENTER to record (or 'q' to quit): ").strip()

        if user_input.lower() == 'q':
            break

        print(f"    🔴 Recording {duration}s... SPEAK NOW!")
        try:
            recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            y = recording.squeeze()

            # Check if there's actual audio (not silence)
            if np.mean(np.abs(y)) < 0.005:
                print("    ⚠️ Too quiet — speak louder! Try again.")
                continue

            if save_audio_wav(y, sr, f"real_mic_{count:04d}.wav"):
                count += 1
                print(f"    ✅ Saved! ({count}/{needed})\n")
            else:
                print("    ❌ Save failed, try again.")

        except Exception as e:
            print(f"    ❌ Recording error: {e}")
            continue

    print(f"    ✅ Recorded {count} microphone samples")
    return count


# ================================================================
# METHOD 8: Bulk Microphone (no prompts, continuous)
# ================================================================
def method_8_bulk_record(needed):
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  Method 8: Bulk Continuous Recording               ║")
    print("  ║  Just talk for a few minutes — we split it up      ║")
    print("  ╚═══════════════════════════════════════════════════╝")

    try:
        import sounddevice as sd
    except ImportError:
        print("    ❌ sounddevice not installed → pip install sounddevice")
        return 0

    sr = 22050
    total_seconds = min(needed * 4, 300)  # ~4s per clip, max 5 minutes

    print(f"\n    🎤 We'll record {total_seconds}s of continuous speech.")
    print(f"    📢 Just TALK NATURALLY for {total_seconds//60}m {total_seconds%60}s.")
    print(f"    📢 Talk about anything — your day, a story, plans, etc.")
    print(f"    📢 Include natural pauses and different emotions.\n")

    user_input = input("    Press ENTER to start (or 'q' to skip): ").strip()
    if user_input.lower() == 'q':
        return 0

    print(f"    🔴 Recording {total_seconds}s... START TALKING!")

    try:
        recording = sd.rec(
            int(total_seconds * sr), samplerate=sr, channels=1, dtype='float32'
        )
        sd.wait()
        y = recording.squeeze()
        print("    ⬜ Recording complete!")

        # Split into 3-7 second segments, skip silence
        count = 0
        seg_len = int(5.0 * sr)  # 5-second segments
        hop = int(3.0 * sr)       # 3-second hop

        for start in range(0, len(y) - seg_len, hop):
            if count >= needed:
                break

            segment = y[start:start + seg_len]

            # Skip silent segments
            energy = np.mean(np.abs(segment))
            if energy < 0.008:
                continue

            if save_audio_wav(segment, sr, f"real_bulk_{count:04d}.wav"):
                count += 1

        print(f"    ✅ Split into {count} voice segments")
        return count

    except Exception as e:
        print(f"    ❌ Recording error: {e}")
        return 0


# ================================================================
# MAIN EXECUTION
# ================================================================
def main():
    print("=" * 65)
    print("  VoiceGuard Pro — Real Human Voice Downloader")
    print("  Target: {} samples in dataset/real/".format(TARGET))
    print("=" * 65)

    existing = count_existing()
    if existing > 0:
        print(f"\n  📁 Found {existing} existing files in dataset/real/")

    if existing >= TARGET:
        print(f"  ✅ Already have enough samples! ({existing}/{TARGET})")
        return

    needed = TARGET - existing
    print(f"  📊 Need {needed} more samples\n")

    total_added = 0

    # ── Try automated methods first ──
    methods = [
        ("Google FLEURS", method_1_fleurs),
        ("LibriSpeech", method_2_librispeech),
        ("VoxPopuli", method_3_voxpopuli),
        ("OpenSLR Direct Download", method_4_direct_download),
        
    ]

    for name, method in methods:
        remaining = TARGET - count_existing()
        if remaining <= 0:
            break

        n = method(remaining)
        total_added += n

        current = count_existing()
        print(f"\n  📊 Progress: {current}/{TARGET} samples")

        if current >= TARGET:
            print(f"  🎉 Target reached!")
            break

    # ── If still not enough, offer interactive recording ──
    current = count_existing()
    if current < TARGET:
        remaining = TARGET - current

        print("\n" + "=" * 65)
        print(f"  ⚠️ Automated methods got {current} samples.")
        print(f"     Still need {remaining} more.")
        print(f"     Let's record from your microphone!")
        print("=" * 65)

        print("\n  Choose recording mode:")
        print("  [1] Guided (prompts for each clip — better variety)")
        print("  [2] Bulk  (talk continuously — we split it up)")
        print("  [3] Skip  (use what we have)")

        choice = input("\n  Enter 1, 2, or 3: ").strip()

        if choice == "1":
            n = method_7_microphone(remaining)
            total_added += n
        elif choice == "2":
            n = method_8_bulk_record(remaining)
            total_added += n
        else:
            print("  ⏭️ Skipping microphone recording")

    # ── Final report ──
    final_count = count_existing()

    print("\n" + "=" * 65)
    print("  📊 FINAL REPORT")
    print("=" * 65)
    print(f"  Total real voice samples: {final_count}")
    print(f"  Location: {os.path.abspath(REAL_DIR)}")

    if final_count >= 100:
        print(f"\n  ✅ EXCELLENT! Ready to train.")
        print(f"     Run: python train_model.py")
    elif final_count >= 50:
        print(f"\n  ✅ GOOD enough to train (100+ recommended).")
        print(f"     Run: python train_model.py")
    elif final_count >= 20:
        print(f"\n  ⚠️ LOW — model will work but accuracy may suffer.")
        print(f"     Consider adding more samples manually.")
    else:
        print(f"\n  ❌ TOO FEW samples for reliable training.")
        print(f"     Please add .wav files manually to: {os.path.abspath(REAL_DIR)}")
        print(f"\n  Quick options:")
        print(f"     • Record yourself + friends/family speaking")
        print(f"     • Download from: https://commonvoice.mozilla.org")
        print(f"     • Download from: https://openslr.org/12/")

    # Verify samples are loadable
    if final_count > 0:
        print(f"\n  🔍 Verifying files...")
        good = 0
        bad = 0
        files = [f for f in os.listdir(REAL_DIR) if f.lower().endswith(EXTENSIONS)][:10]
        for f in files:
            try:
                import librosa
                y, sr = librosa.load(os.path.join(REAL_DIR, f), sr=22050, duration=3)
                if len(y) > 0:
                    good += 1
                else:
                    bad += 1
            except:
                bad += 1

        if bad == 0:
            print(f"  ✅ All {good} checked files load correctly")
        else:
            print(f"  ⚠️ {bad}/{good+bad} files have issues")

    print("=" * 65)


if __name__ == "__main__":
    main()