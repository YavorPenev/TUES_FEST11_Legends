import React, { useEffect, useState } from "react";
import Header from "./assets/header";
import Footer from "./assets/footer";
import { fetchNotes } from "./network/index";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const data1 = [
  { name: "Jan", value: 40 },
  { name: "Feb", value: 60 },
  { name: "Mar", value: 80 },
  { name: "Apr", value: 50 },
];

const data2 = [
  { name: "Jan", value: 20 },
  { name: "Feb", value: 30 },
  { name: "Mar", value: 70 },
  { name: "Apr", value: 90 },
];

const Dashboard = () => {
  const [diagrams, setDiagrams] = useState([
    { id: 1, text: "Diagrama 1" },
    { id: 2, text: "Diagrama 2" },
    { id: 3, text: "Diagrama 3" },
    { id: 4, text: "Diagrama 4" },
    { id: 5, text: "Diagrama 5" },
    { id: 6, text: "Diagrama 6" },
    { id: 7, text: "Diagrama 7" },
    { id: 8, text: "Diagrama 8" },
    { id: 9, text: "Diagrama 9" },
    { id: 10, text: "Diagrama 10" },
    { id: 11, text: "Diagrama 11" },
    { id: 12, text: "Diagrama 12" },
  ]);

  const [notes, setNotes] = useState([
    { id: 1, title: "Note 1", text: "Text 1." },
    { id: 2, title: "Note 2", text: "text 2." },
    { id: 3, title: "Note 3", text: "text 3." },
    { id: 4, title: "Note 4", text: "text 4." },
    { id: 5, title: "Note 5", text: "text 5." },
    { id: 6, title: "Note 6", text: "text 6." },
    { id: 7, title: "Note 7", text: "text 7." },
    { id: 8, title: "Note 8", text: "text 8." },
    { id: 9, title: "Note 9", text: "text 9." },
    { id: 10, title: "Note 10", text: "text 10." },
    { id: 11, title: "Note 11", text: "text 11." },
    { id: 12, title: "Note 12", text: "text 12." },
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

  const deleteDiagram = (id) => {
    setDiagrams((prev) => prev.filter((d) => d.id !== id));
  };


  return (
    <div className="bg-gray-100 h-screen flex flex-col">
      <Header />
      <div className="flex h-screen overflow-hidden font-sans mt-24">
        <div className="w-75 bg-gray-900 text-white flex-shrink-0 p-6 space-y-6">
          <div className="flex justify-center">
            <div className="w-40 h-40 bg-white text-black flex items-center justify-center text-sm rounded-full">
              <img src="../public/proficon.jpg" alt="Profile" className="rounded-full w-full h-full object-cover" />
            </div>
          </div>
          <p className="text-center text-3xl font-bold">Dimcho Dimov</p>
          <div className="space-y-7 gap-2 mt-25 ">
            <button className="w-full bg-gray-700 hover:bg-gray-600 py-2 rounded">AI Advisor</button>
            <button className="w-full bg-gray-700 hover:bg-gray-600 py-2 rounded">Stock Advisor</button>
            <button className="w-full bg-gray-700 hover:bg-gray-600 py-2 rounded">Budget Advisor</button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto bg-gray-100 p-6">
          <h2 className="text-xl font-semibold mb-4">Stocks Advisor Diagrams</h2>
          <div className="space-y-6 h-75">
            {diagrams.map((diagram) => (
              <div key={diagram.id} className="bg-white rounded-xl shadow p-4">
                <div className="flex justify-between">
                  <button
                    onClick={() => deleteDiagram(diagram.id)}
                    className="text-red-500 hover:underline font-bold text-lg "
                  >
                    ✕
                  </button>
                </div>

       
                <div className="bg-black rounded-xl mt-2 p-4 h-[20hv]">
                  <ResponsiveContainer width="100%" height={100}>
                    <LineChart data={diagram.id === 1 ? data1 : data2}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                      <XAxis dataKey="name" stroke="#aaa" />
                      <YAxis stroke="#aaa" />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#00FF00"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="flex justify-between items-center mt-4">
                  <span className="font-semibold text-lg text-center">Diagram {diagram.id}</span>
                  <button className="bg-blue-600 text-white rounded hover:bg-blue-500 py-2 px-4">
                    View
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <h2 className="text-xl font-semibold mt-8 mb-4">Notes</h2>
        <div className="space-y-6">
          {notes.map((note) => (

            <div key={note.id} className="bg-blue-100 p-4 rounded-xl shadow w-120"  >
              <div className="flex justify-between">
                <span className="font-medium text-blue-900 text-lg">
                  {note.title}
                </span>
              </div>
              <p className="text-blue-800 text-lg mt-2 border-t border-dashed pt-2">
                {note.content}
              </p>
            </div>
          ))}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Dashboard;