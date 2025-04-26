import { useState, useRef, React, useEffect } from 'react';
import { Routes, Route, useNavigate, Link, useLocation } from 'react-router';
import './styles/index.css';
import Header from './assets/header';
import Footer from './assets/footer';
import TopCompanies from './assets/topcompanies';
import ArticleCarousel from './assets/articlecarousel';
import News from './articles';
import UsefulSources from './assets/usefulsources';
import ProfArticles from './assets/professionalarticles';
import axios from "axios";
import ArticleDetails from "./assets/ArticleDetails";

function Home() {
  const [selectedArticle, setSelectedArticle] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  const aimenuref = useRef(null);

  const [AiStatus, SetAiStatus] = useState(false);

  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchNews = async () => {
    try {
      const response = await axios.get("http://localhost:8000/api/news");
      if (response.data.success) {
        const formattedArticles = response.data.data.map((article) => ({
          id: article.uuid,
          title: article.title,
          image: article.image_url,
          description: article.description,
          content: article.content,
          published_at: article.published_at,
          source: article.source,
          url: article.url
        }));
        setArticles(formattedArticles);
      }
    } catch (err) {
      console.error("Error fetching articles:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNews();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  const AiMenuChange = () => {
    SetAiStatus(!AiStatus);
    if (aimenuref.current) {
      aimenuref.current.style.display = AiStatus ? 'none' : 'flex';
    }
  };

  //pt-24 - na wsqka stranica za da raboti hedyra
  const hideHeader = location.pathname === '/login' || location.pathname === '/signup';

  if (selectedArticle) {
    return (
      <>
        <Header onLogoClick={() => setSelectedArticle(null)} />
        <div className="bg-blue-100 h-full m-0 pt-28 pb-28">
          <ArticleDetails
            article={selectedArticle}
            onBack={() => setSelectedArticle(null)}
          />
        </div>
        <Footer />
      </>
    );
  }

  return (
    <>
      <Header />

      <div className="pt-20">
        <div
          ref={aimenuref}
          className="z-40 flex-col-reverse rounded-t-full fixed right-5 bottom-17 pb-15 pt-10 gap-4 text-blue-100 w-25 bg-blue-800 border-4 border-blue-950"
          style={{ display: 'none' }}
        >
          <Link
            className="w-[88%] ml-[6%] mr-[6%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
            to="/invest"
          >
            Stocks<br />
            Advisor
          </Link>
          
          <Link
            className="w-[88%] ml-[6%] mr-[6%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
            to="/advice"
          >
            Investment<br />
            Advisor
          </Link>
          <Link
            className="w-[88%] ml-[6%] mr-[6%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
            to="/budgetplanner"
          >
            Budget<br />
            Planner
          </Link>
        </div>


        <button
          className="z-50 w-25 h-25 text-3xl pb-23 font-extrabold rounded-full border-blue-950 border-4 bg-blue-800 text-blue-100 fixed bottom-5 right-5 cursor-pointer hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
          onClick={AiMenuChange}
        >
          <sub>^</sub>
          <br />
          <b>AI</b>
        </button>

        <img
          className="-mt-28 w-full border-b-8 border-b-blue-900"
          src="/mainpic-above.png"
          alt="Main Visual"
        />

        <div className="flex justify-center items-center pb-5 pt-5 w-full flex-col bg-gradient-to-b from-blue-100 to-gray-400 border-b-8 border-blue-900">
          <img className="w-[30%]" src="/SmartBudget.png" alt="SmartBudget Logo" />
          <Link
            to="/faq"
            className="absolute top-[39%] right-4 bg-blue-900 pt-2 pb-2 pr-5 pl-5 rounded-2xl text-2xl font-extrabold text-blue-100 hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
          >
            ?
          </Link>

          <h1 className="text-4xl font-semibold text-blue-900">
            Your Ultimate Financial Knowledge Hub
          </h1>
          <Link
            to="/learnmore"
            className="bg-blue-800 mt-10 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-2xl text-blue-100 font-bold hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
          >
            Learn More
          </Link>
        </div>

        <img className="w-full" src="/mainpic-bottom.png" alt="Footer Visual" />
      </div>

      <ArticleCarousel articles={articles} onArticleSelect={setSelectedArticle} />

      {selectedArticle && (
        <div className="bg-blue-50 p-10 border-t-8 border-blue-900">
          <h2 className="text-3xl font-bold text-blue-900 mb-4">{selectedArticle.title}</h2>
          <img src={selectedArticle.image} alt={selectedArticle.title} className="w-full max-h-96 object-cover rounded mb-6" />
          <p className="text-lg text-gray-800 mb-4">{selectedArticle.content}</p>
          <a href={selectedArticle.url} target="_blank" rel="noopener noreferrer"
            className="text-blue-700 font-semibold underline"
          >
            Read more at {selectedArticle.source}
          </a>
          <button
            onClick={() => setSelectedArticle(null)}
            className="block mt-6 bg-blue-800 text-white font-bold py-2 px-4 rounded hover:scale-105 transition-transform"
          >
            Back to Articles
          </button>
        </div>
      )}


      <div className="bg-[url('/Background_Sections.png')] bg-cover bg-center bg-no-repeat p-10 border-t">
        <div className="relative z-10">
          <ProfArticles />
          <UsefulSources />
        </div>
      </div>
      <TopCompanies />
      <Footer />
    </>

  );


}

export default Home;