import React, { useState } from 'react';
import { login1 } from "./network/index";

function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async () => {
    try {
      const user = await login1(username, password);
      alert(`Welcome back, ${user.username}!`);
      console.log("User data:", user);
  
      // Съхранете информацията за потребителя в localStorage
      localStorage.setItem("user", JSON.stringify(user));
  
      // Пренасочване към началната страница
      window.location.href = "/";
    } catch (error) {
      alert(`Login failed: ${error.response?.data?.error || error.message}`);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-b from-blue-100 to-blue-300">
      <div className="bg-white shadow-lg rounded-xl p-15 w-[90%] max-w-md">
        <h1 className="text-3xl font-bold text-center text-blue-800 mb-6">Welcome Back!</h1>
        <p className="text-center text-gray-600 mb-8">Please log in to your account</p>
        
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
            className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
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
            className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <button
          type="button"
          id="login-btn"
          onClick={handleLogin}
          className="w-full bg-blue-600 text-white font-bold py-3 rounded-lg hover:bg-blue-700 transition duration-300"
        >
          Login
        </button>
        
        <p className="text-center text-gray-600 mt-6">
          Don't have an account?{" "}
          <a href="/signup" className="text-blue-600 font-semibold hover:underline">
            Sign up
          </a>
        </p>
      </div>
    </div>
  );
}

export default Login;