import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Header from './assets/header';
import Footer from './assets/footer';

function ArticleDetails() {
  const location = useLocation();
  const navigate = useNavigate();
  const article = location.state;

  if (!article) {
    // If no article data is found, redirect back to the articles page
    navigate("/articles");
    return null;
  }

  return (
    <>
      <Header />
      <div className="min-h-screen bg-gray-100 text-gray-800 p-6 mt-25 flex flex-row ">
        <div className="flex justify-between items-center mb-6 h-16 -mr-10">
          <button
            onClick={() => navigate(-1)}
            className="bg-blue-800 text-white px-4 py-2 rounded-lg hover:bg-blue-900 transition duration-200 cursor-pointer"
          >
            ← Back
          </button>
        </div>
        <div className="max-w-3xl mx-auto bg-white rounded-2xl shadow-lg p-6 flex flex-col gap-4">
          <img
            src={article.image}
            className="w-full h-64 object-fill rounded-lg"
            alt={article.title}
          />
          <h2 className="text-2xl font-bold">{article.title}</h2>
          <p className="text-gray-600">{article.discription}</p>
        </div>
      </div>
      <Footer />
    </>
  );
}

export default ArticleDetails;