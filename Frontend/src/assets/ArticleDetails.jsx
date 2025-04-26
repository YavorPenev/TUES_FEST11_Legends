import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Header from './header';
import Footer from './footer';

function ArticleDetails({ article, onBack }) {
  if (!article) return null;

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg max-w-4xl mx-auto mt-0">
      <button
        onClick={onBack}
        className="text-blue-600 hover:underline mb-4 flex items-center"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
        </svg>
        Back to Main Page
      </button>

      <h2 className="text-2xl font-bold mb-4">{article.title}</h2>

      <div className="flex items-center text-sm text-gray-500 mb-4">
        <span>{article.source}</span>
        <span className="mx-2">•</span>
        <span>{new Date(article.published_at).toLocaleDateString()}</span>
      </div>

      {article.image_url || article.image ? (
        <img
          src={article.image_url || article.image}
          alt={article.title}
          className="w-full h-64 object-cover rounded-lg mb-4"
          onError={(e) => {
            e.target.src = 'https://via.placeholder.com/800x400?text=News+Image';
          }}
        />
      ) : null}

      {article.description && <p className="text-gray-700 mb-4">{article.description}</p>}

      {article.content && (
        <p className="text-gray-600 mb-6">
          {article.content.replace(/\[\+\d+ chars\]/, '')}
        </p>
      )}

      <a
        href={article.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center text-blue-600 hover:underline"
      >
        Read Full Article
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
        </svg>
      </a>
    </div>
  
  );
}

export default ArticleDetails;