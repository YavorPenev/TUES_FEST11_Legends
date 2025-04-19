import React, { useState } from "react";
import { Link } from "react-router";
import { ChevronLeft, ChevronRight } from "lucide-react";

const ArticleCarousel = ({ articles }) => {
  const [currentSlide, setCurrentSlide] = useState(0);

  const groupedArticles = [];
  for (let i = 0; i < Math.floor(articles.length / 3) * 3; i += 3) {
    groupedArticles.push(articles.slice(i, i + 3));
  }

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % groupedArticles.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + groupedArticles.length) % groupedArticles.length);
  };

  return (
    <div className="bg-blue-950 text-white py-10">
      <div className="flex items-center justify-between px-10">
        <button onClick={prevSlide}>
          <ChevronLeft className="text-blue-950 scale-160 w-6 h-6 hover:scale-180 transition-transform active:scale-140 bg-blue-100 p-1 rounded text-xl active:duration-50" />
        </button>
        <h1 className="text-2xl font-bold text-center flex-grow">Featured Articles</h1>
        <button onClick={nextSlide}>
          <ChevronRight className="text-blue-950 scale-160 w-6 h-6 hover:scale-180 transition-transform active:scale-140 bg-blue-100 p-1 rounded text-xl active:duration-50" />
        </button>
      </div>

      <div className="mt-6 px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {groupedArticles[currentSlide].map((article) => (
            <Link
              key={article.id}
              to="/article"
              state={article}
              className="cursor-pointer rounded-2xl overflow-hidden shadow-md hover:shadow-xl transition transform hover:scale-105"
            >
              <div className="bg-white text-black">
                <img
                  src={article.image}
                  className="w-full h-40 object-cover"
                  alt={article.title}
                />
                <div className="p-4 h-32 flex flex-col justify-center">
                  <h3 className="text-xl font-semibold text-center line-clamp-3">
                    {article.title}
                  </h3>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ArticleCarousel;