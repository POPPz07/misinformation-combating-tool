// src/firebase.js

import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut } from "firebase/auth";

// TODO: Replace this with your app's actual Firebase config
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDDESy3idoTe5N6P2eIZFilsp0f_ni6R80",
  authDomain: "flash-clover-476914-h6.firebaseapp.com",
  projectId: "flash-clover-476914-h6",
  storageBucket: "flash-clover-476914-h6.firebasestorage.app",
  messagingSenderId: "961233506701",
  appId: "1:961233506701:web:d3792a0a6dbabc9d83edc4",
  measurementId: "G-840R46HZ92"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

export { signInWithPopup, signOut };