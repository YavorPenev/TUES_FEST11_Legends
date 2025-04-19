import React, { useState } from 'react';
import { signup1 } from "./network/index";

function SignUP() {
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSignup = async () => {
    try {
      const message = await signup1(username, email, password);
      alert(message); 
      console.log("Signup successful:", { username, email });
    } catch (error) {
      alert(`Signup failed: ${error.response?.data?.error || error.message}`);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-b from-blue-100 to-blue-300">
      <div className="bg-white shadow-lg rounded-2xl p-10 w-[90%] max-w-md">
        <h1 className="text-3xl font-bold text-center text-blue-800 mb-6">Create Your Account</h1>
        <p className="text-center text-gray-600 mb-8">Join us and start your journey!</p>

        <div className="mb-4">
          <label htmlFor="email" className="block text-gray-700 font-semibold mb-2">
            Email
          </label>
          <input
            type="email"
            id="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="mb-4">
          <label htmlFor="username" className="block text-gray-700 font-semibold mb-2">
            Username
          </label>
          <input
            type="text"
            id="username"
            placeholder="Enter your username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="mb-6">
          <label htmlFor="password" className="block text-gray-700 font-semibold mb-2">
            Password
          </label>
          <input
            type="password"
            id="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <button
          type="button"
          id="signup-btn"
          onClick={handleSignup}
          className="w-full bg-blue-600 text-white font-bold py-3 rounded-2xl hover:bg-blue-700 transition duration-300"
        >
          Sign Up
        </button>

        <p className="text-center text-gray-600 mt-6">
          Already have an account?{' '}
          <a href="/login" className="text-blue-600 font-semibold hover:underline">
            Log in
          </a>
        </p>
      </div>
    </div>
  );
}

export default SignUP;