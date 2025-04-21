import React, { useEffect, useState, useRef } from "react";
import Header from "./assets/header";
import { fetchNotes } from "./network/index";
import axios from "axios";
import { Link } from "react-router";

const Dashboard = () => {
  const user = JSON.parse(localStorage.getItem("user"));
  const username = user?.username || "Guest";

  const responseRef = useRef(null);
  const [responseHeight, setResponseHeight] = useState("auto");

  const [notes, setNotes] = useState([
    { id: 1, title: "Note 1", content: "Text 1." },
    { id: 2, title: "Note 2", content: "Text 2." },
    { id: 3, title: "Note 3", content: "Text 3." },
    { id: 4, title: "Note 4", content: "Text 4." },
    { id: 5, title: "Note 5", content: "Text 5." },
    { id: 6, title: "Note 6", content: "Text 6." },
    { id: 7, title: "Note 7", content: "Text 7." },
    { id: 8, title: "Note 8", content: "Text 8." },
    { id: 9, title: "Note 9", content: "Text 9." },
    { id: 10, title: "Note 10", content: "Text 10." },
    { id: 11, title: "Note 11", content: "Text 11." },
    { id: 12, title: "Note 12", content: "Text 12." },
  ]);

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

  useEffect(() => {
    if (responseRef.current) {
      setResponseHeight(`${responseRef.current.offsetHeight + 200}px`);
    }
  }, [responses, expandedResponseId]);

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
    <div className="bg-gray-100 flex flex-col">
      <div className="bg-gray-100 flex flex-col">
        <Header />
        <div className="flex overflow-hidden font-sans mt-24">
          {/* Sidebar */}
          <div className="w-75 bg-gray-800 text-white flex-shrink-0 space-y-6">
            <div className="flex items-center justify-center mb-4 bg-gray-800 p-4 h-1/2 flex-col w-full">
              <div className="flex justify-center">
                <div className="w-40 h-40 bg-white text-black flex items-center justify-center text-sm rounded-full mt-10 overflow-hidden">
                  <img src="../public/proficon.jpg" alt="Profile" className="rounded-full w-full h-full object-cover" />
                </div>
              </div>
              <p className="text-center text-3xl font-bold mt-6">{username}</p>
            </div>
          </div>

          <div className="flex flex-col items-center mt-24 w-full">
            <h1 className="text-4xl font-bold text-blue-800 mb-6">Here are your saved responses,  {username}</h1>

            {/* Buttons for AI Models */}
            <div className="flex space-x-4 mb-6">
              <button
                onClick={() => handleModelSelection("stocks")}
                className={`px-6 py-2 rounded-lg font-bold ${selectedModel === "stocks" ? "bg-blue-800 text-white" : "bg-gray-300 text-gray-800"
                  }`}
              >
                Stocks Advisor
              </button>
              <button
                onClick={() => handleModelSelection("investment")}
                className={`px-6 py-2 rounded-lg font-bold ${selectedModel === "investment" ? "bg-blue-800 text-white" : "bg-gray-300 text-gray-800"
                  }`}
              >
                Investment Advisor
              </button>
              <button
                onClick={() => handleModelSelection("budget")}
                className={`px-6 py-2 rounded-lg font-bold ${selectedModel === "budget" ? "bg-blue-800 text-white" : "bg-gray-300 text-gray-800"
                  }`}
              >
                Budget Planner
              </button>
            </div>

            {/* Responses Section */}
            <div
              ref={responseRef}
              className="w-full max-w-4xl bg-white rounded-lg shadow-md p-6 overflow-y-auto scrollbar-thin scrollbar-thumb-blue-400 scrollbar-track-blue-100"
            >
              {responses.length === 0 ? (
                <p className="text-gray-600 text-center">No responses available for this model.</p>
              ) : (
                responses.map((response) => (
                  <div key={response.id} className="mb-4 border-b pb-4">
                    {/* Summary */}
                    <div className="flex justify-between items-center">
                      <p className="text-gray-800 font-semibold">
                        {response.summary || response.fullResponse.split("\n")[0]}
                      </p>
                      <button
                        onClick={() => toggleExpandResponse(response.id)}
                        className="text-blue-600 hover:underline"
                      >
                        {expandedResponseId === response.id ? "Collapse" : "Expand"}
                      </button>
                    </div>

                    {/* Full Response */}
                    {expandedResponseId === response.id && (
                      <div className="mt-2 bg-gray-100 p-4 rounded-lg">
                        <pre className="whitespace-pre-wrap text-gray-700 text-xs">
                          {response.fullResponse}
                        </pre>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Notes */}
          <div className="flex flex-col items-center mt-0 w-full">
            <h1 className="text-3xl font-bold text-blue-800 mb-4">Notes</h1>
            <div
              className="mt-4 w-full max-w-4xl bg-gray-800 rounded-lg shadow-md p-6 overflow-y-auto scrollbar-thin scrollbar-thumb-blue-400 scrollbar-track-blue-100"
              style={{ height: responseHeight }}
            >

              <div className="space-y-6">
                {notes.map((note) => (
                  <div key={note.id} className="bg-blue-50 p-4 rounded-xl shadow">
                    <div className="flex justify-between">
                      <span className="font-semibold text-blue-900 text-lg">{note.title}</span>
                    </div>
                    <p className="text-blue-800 text-lg mt-2 border-t border-dashed pt-2">
                      {note.content}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>

        </div>
      </div>

      <footer className="flex bg-blue-800 bg-gradient-to-b from-transparent to-gray-800 text-blue-100 p-5 justify-evenly flex-wrap items-center border-t-8 border-t-blue-900 w-full">
        <div className="flex-col flex-nowrap justify-evenly gap-2 items-center justify-items-center">
          <a
            href="https://www.youtube.com/watch?v=MpxpUVjfFaE"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img className="aspect-auto h-15" src="/youtube.png" alt="YouTube" />
          </a>
          <p>Copyright @2025</p>
          <p>©legends Development Team</p>
        </div>
        <div className="flex-col flex-nowrap justify-evenly gap-2 items-center justify-items-center">
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
