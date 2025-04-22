import React, { useEffect, useState, useRef } from "react";
import Header from "./assets/header";
import { fetchNotes } from "./network/index";
import axios from "axios";
import { Link } from "react-router";

const Dashboard = () => {
  const user = JSON.parse(localStorage.getItem("user"));
  const username = user?.username || "Guest";

  const responseRef = useRef(null);
  const [notes, setNotes] = useState([]);

  useEffect(() => {
    const loadNotes = async () => {
      try {
        const data = await fetchNotes();
        console.log("Fetched notes:", data);
        setNotes(data);
      } catch (error) {
        console.error("Failed to fetch notes:", error);
      }
    };

    loadNotes();
  }, []);

  const [selectedModel, setSelectedModel] = useState("stocks");
  const [responses, setResponses] = useState([]);
  const [expandedResponseId, setExpandedResponseId] = useState(null);

  const fetchResponses = async (model) => {
    try {
      const endpointMap = {
        stocks: "/get-stock-advice",
        investment: "/get-investment-advice",
        budget: "/get-budget-planner",
      };
      const response = await axios.get(`http://localhost:8000${endpointMap[model]}`, {
        withCredentials: true,
      });
      setResponses(response.data.responses || []);
    } catch (error) {
      console.error("Error fetching responses:", error);
      setResponses([]);
    }
  };

  const handleModelSelection = (model) => {
    setSelectedModel(model);
    setExpandedResponseId(null);
    fetchResponses(model);
  };

  const toggleExpandResponse = (id) => {
    setExpandedResponseId((prevId) => (prevId === id ? null : id));
  };

  useEffect(() => {
    fetchResponses(selectedModel);
  }, [selectedModel]);

  return (
    <div className="flex flex-col min-h-screen bg-gray-100">
      <Header />
      <div className="flex flex-1 overflow-hidden mt-24">
   
        <div className="w-74 bg-gray-800 text-white flex-shrink-0 flex flex-col items-center py-6 space-y-6 overflow-y-auto">
          <div className="w-40 h-40 bg-white text-black flex items-center justify-center text-sm rounded-full overflow-hidden">
            <img src="../public/proficon.jpg" alt="Profile" className="rounded-full w-full h-full object-cover" />
          </div>
          <p className="text-center text-3xl font-bold">{username}</p>
        </div>

        <div className="flex flex-col w-2/3 h-[calc(100vh-6rem)] overflow-hidden p-6 space-y-6">
          <h1 className="text-4xl font-bold text-blue-800">Saved responses, {username}</h1>
          <div className="flex space-x-4">
            <button onClick={() => handleModelSelection("stocks")} className={`px-6 py-2 rounded-lg font-bold ${selectedModel === "stocks" ? "bg-blue-800 text-white" : "bg-gray-300 text-gray-800"}`}>Investment Advisor</button>
            <button onClick={() => handleModelSelection("investment")} className={`px-6 py-2 rounded-lg font-bold ${selectedModel === "investment" ? "bg-blue-800 text-white" : "bg-gray-300 text-gray-800"}`}>Stocks Advisor</button>
            <button onClick={() => handleModelSelection("budget")} className={`px-6 py-2 rounded-lg font-bold ${selectedModel === "budget" ? "bg-blue-800 text-white" : "bg-gray-300 text-gray-800"}`}>Budget Planner</button>
          </div>

          <div ref={responseRef} className="bg-white rounded-lg shadow-md p-6 overflow-y-auto scrollbar-thin scrollbar-thumb-blue-400 scrollbar-track-blue-100 flex-1">
            {responses.length === 0 ? (
              <p className="text-gray-600 text-center text-lg">No responses available for this model.</p>
            ) : (
              responses.map((response) => (
                <div key={response.id} className="mb-4 border-b pb-4">
                  <div className="flex justify-between items-center">
                    <p className="text-gray-800 font-semibold text-lg">{response.summary || response.fullResponse.split("\n")[0]}</p>
                    <button onClick={() => toggleExpandResponse(response.id)} className="text-blue-600 hover:underline text-base">
                      {expandedResponseId === response.id ? "Collapse" : "Expand"}
                    </button>
                  </div>
                  {expandedResponseId === response.id && (
                    <div className="mt-2 bg-gray-100 p-4 rounded-lg">
                      <pre className="whitespace-pre-wrap text-gray-700 text-base">{response.fullResponse}</pre>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
   
        <div className="bg-gray-800 text-white flex-shrink-0 p-4 w-2/6 h-[calc(100vh-6rem)] flex flex-col">
          <h2 className="text-3xl font-semibold mt-2 mb-4">Notes</h2>
          <div className="space-y-6 overflow-y-auto scrollbar-thin scrollbar-thumb-blue-500 scrollbar-track-blue-300 flex-1 pr-1">
            {notes.map((note) => (
              <div key={note.id} className="bg-gray-700 p-4 rounded-xl shadow">
                <div className="flex justify-between">
                  <span className="font-medium text-blue-300 text-lg">{note.title}</span>
                </div>
                <p className="text-blue-200 text-base mt-2 border-t border-dashed border-blue-500 pt-2">
                  {note.content}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <footer className="flex bg-blue-800 bg-gradient-to-b from-transparent to-gray-800 text-blue-100 p-5 justify-evenly flex-wrap items-center border-t-8 border-t-blue-900 w-full">
        <div className="flex-col flex-nowrap justify-evenly gap-2 items-center">
          <a href="https://www.youtube.com/watch?v=MpxpUVjfFaE" target="_blank" rel="noopener noreferrer">
            <img className="aspect-auto h-15" src="/youtube.png" alt="YouTube" />
          </a>
          <p>Copyright @2025</p>
          <p>©legends Development Team</p>
        </div>
        <div className="flex-col flex-nowrap justify-evenly gap-2 items-center">
          <p>legends@gmail.com</p>
          <p>+39 06 6988 4857</p>
          <p>+39 04 5355 9832</p>
        </div>
        <div className="flex flex-col gap-1 items-center">
          <Link to="/investcalc" className="text-blue-100 hover:underline">Investment Calculator</Link>
          <Link to="/calcloan" className="text-blue-100 hover:underline">Loan Calculator</Link>
          <Link to="/CurrencyCalculator" className="text-blue-100 hover:underline">Currency Calculator</Link>
        </div>
        <div className="flex flex-col gap-1 items-start">
          <Link to="/about" className="text-blue-100 hover:underline">About Us</Link>
          <Link to="/articles" className="text-blue-100 hover:underline">Latest News</Link>
          <Link to="/usefulsources" className="text-blue-100 hover:underline">Useful Sources</Link>
        </div>
      </footer>
    </div>
  );
};

export default Dashboard;