# AI Dental Studio

Mobile app that analyze dental radiographs (OPG X-rays) using advanced AI models.

## Contents

1. [Flutter Project Setup Guide](#flutter-project-setup-guide)
   - [Prerequisites](#prerequisites)
   - [Installation Steps](#installation-steps)
   - [Running the Project](#running-the-project)
   
2. [AI-Dental Studio: Beta Testing Installation Guide](#ai-dental-studio-beta-testing-installation-guide)
   - [iOS Installation (via TestFlight)](#ios-installation-via-testflight)
   - [Android Installation (via Firebase App Distribution)](#android-installation-via-firebase-app-distribution)


## Getting Started
# Flutter Project Setup Guide

This guide will help you set up the Flutter development environment and run the project on macOS.

## Prerequisites

- macOS (for iOS development)
- Terminal access
- Admin privileges

---

## Installation Steps

### 1. Download Flutter SDK

Download the Flutter SDK from the official Flutter website:

[Download Flutter SDK](https://docs.flutter.dev/install/archive)

### 2. Extract Flutter SDK

Unzip the downloaded Flutter SDK to a directory of your choice:
```bash
cd ~/development
unzip ~/Downloads/flutter_macos_*.zip
```

Note the path to the `flutter/bin` directory (e.g., `~/development/flutter/bin`)

### 3. Set Environment Variable PATH

Add Flutter to your PATH environment variable:
```bash
# Open your shell configuration file
nano ~/.zshrc
```

Add the following line (replace `/path-to/flutter/bin` with your actual Flutter bin path):
```bash
export PATH="$PATH:/Users/yourusername/development/flutter/bin"
```

Save and exit:
- Press `Ctrl + X`
- Press `Y` to confirm
- Press `Enter` to save

Apply the changes:
```bash
source ~/.zshrc
```

### 4. Verify Flutter Installation

Check if Flutter is correctly installed:
```bash
flutter --version
```

If you see the Flutter version information, the PATH is set correctly.

### 5. Install Xcode (macOS Only)

1. Open the **App Store**
2. Search for **Xcode**
3. Click **Install**

### 6. Accept Xcode License

After Xcode installation, accept the license agreement:
```bash
sudo xcodebuild -license accept
```

Enter your password when prompted.

### 7. Configure Android SDK

Accept Android licenses:
```bash
flutter doctor --android-licenses
```

Press `y` to accept each license when prompted.

### 8. Verify Installation

Run Flutter Doctor to check your setup:
```bash
flutter doctor
```

This will show you if there are any missing dependencies.

---

## Running the Project

### 1. Start iOS Simulator

Open the iOS Simulator:
```bash
open -a Simulator
```

### 2. Clone and Setup Project

Clone the project repository and navigate to the root directory:
```bash
git clone https://cseegit.essex.ac.uk/24-25-ce901-sl-ce902-su/24-25_CE901-SL_CE902-SU_babar_muhammad.git
cd dissertation_code/essex_dental_cleaning/
```

Initialize the project:
```bash
make init
```

### 3. Run the Application

#### Option A: Run on Specific Simulator (Using Simulator ID)

Get list of available devices:
```bash
flutter devices
```

Run on specific device using its ID:
```bash
flutter run -d example-id-48F9A5A0-1DD6-4F29-9EE3-30A71243C248
```

#### Option B: Run on iOS Platform
```bash
flutter run -d ios
```

#### Option C: Run on Android Platform
```bash
flutter run -d android
```

### 4. Check Available Devices

To see all available devices and their IDs:
```bash
flutter devices
```

# AI-Dental Studio: Beta Testing Installation Guide

This guide will help you install the AI-Dental Studio app for beta testing on both iOS and Android devices.

---

## iOS Installation (via TestFlight)

### Prerequisites

- iPhone or iPad running **iOS 13.0 or later**
- Apple ID signed into the App Store

### Installation Steps

#### 1. Open the TestFlight Invitation Link

Tap this link on your iOS device:

**[Join Beta via TestFlight](https://testflight.apple.com/join/pH9TB7UE)**

#### 2. Install TestFlight (if needed)

- If TestFlight is **not installed**, you will be redirected to the App Store
- Tap **Get** to download TestFlight
- Once TestFlight is installed, return to the invitation link above

#### 3. Accept the Invitation

- The link will open in TestFlight
- Tap **Accept** to join the AI-Dental Studio beta program

#### 4. Install the App

- Tap **Install** to download the app on your device
- The app will appear on your home screen once installation is complete

#### 5. Launch the App

- Tap the **AI-Dental Studio** icon on your home screen to launch it
- You are ready to start testing!

---

## Android Installation (via Firebase App Distribution)

### Prerequisites

- Android device with **version 5.0 (API 23) or later**
- Google account

### Installation Steps

#### 1. Open the Firebase Distribution Link

Tap this link on your Android device:

**[Download Beta via Firebase](https://appdistribution.firebase.dev/i/9d9170c6a3831c0d)**

#### 2. Sign in With Your Google Account

- You will be prompted to sign in with your Google account
- Use the email address that was invited to the beta program

#### 3. Install App Tester (if needed)

- Firebase may ask you to install the **"App Tester"** app
- Tap **Install** to download it from the Play Store
- Open **App Tester** and sign in with your Google account

#### 4. Allow Installation from Unknown Sources

When you try to install the app, Android may block the installation. Follow these steps:

1. Tap **Settings** when prompted
2. Enable **Allow from this source**
3. Go back and tap **Install** again

#### 5. Install AI-Dental Studio

- After allowing unknown sources, tap **Install**
- Wait for the installation to complete

#### 6. Launch the App

- Once installed, open the app from your home screen
- You are ready to start testing.
