import React, { useState } from 'react';
import Header from './assets/header';
import Footer from './assets/header';

function FAQ() {
  const [activeIndex, setActiveIndex] = useState(null);

  const toggleAnswer = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  const questions = [
    {
      question: "How to create an account in the site?",
      answer:
        "To create an account on the website, click on the 'Sign Up' or 'Create Account' button, usually located at the top of the homepage. You will be prompted to provide your username, email address, and a password. Some websites may also require you to confirm your email address by clicking a verification link sent to your inbox.",
    },
    {
      question: "How do I log in to my account?",
      answer:
        "To log in, click on the 'Login' button at the top-right corner of the homepage. Enter your registered email address and password, then click 'Submit'. If you’ve forgotten your password, use the 'Forgot Password' option to reset it.",
    },
    {
      question: "How to use the calculators?",
      answer:
        "On nearly every page with a calculator on the left on your screen you can see very concise and useful descrioption of how to use the calculator. The calculators with discription of how to us them are the loan and stock calculators.",
    },
    {
      question: "How to use our AI pages?",
      answer:
        "Exactly like the calculators you can see a very concise and useful descrioption of how to use the AI pages on the left on your screen. The AI pages with discription of how to us them are the Invest advisor and Stock advisor.",
    },
    {
      question: "How to add your notes?",
      answer:
        "To add a note to your profile you need to be logged in. After that you can see a button on all of the calcilators and AI pages. Click on the button and a pop up will appear. Fill in the title and the text of your note and click on the save button.",
    },
    {
      question: "How to see your notes?",
      answer:
        "To see your notes you need to be logged in too. After that you need to click on the profile button on the top of page. After that you will be redirected to your profile page where you can see all of your notes.",
    },
    {
      question: "How to write an usefull source? * ",
      answer:
        "To write an usefull source on the site, you need to click on the + button in the first section on the home page and follow the instructions.",
    },
    {
      question: "How to add links to usefull sites ? * ",
      answer:
        "To add links to your usefull source, you need to click on the + button in the first section on the home page, then you need to copy the link you want to put in and finnaly to paste it on the url section .",
    },
    {
      question: "How to learn more about the site?",
      answer:
        "To learn more about our site and what can you do in it you need to click on the Learn more section in the home page. After that you will be redirected to the page where you can see all of the features of our site.",
    },
    {
      question: "How to see more about us?",
      answer:
        "To see more about us you need to click on the About us section in the home page. After that you will be redirected to the page where you can see all of the information about us.",

    },
    {
      question: "Is our site safe to use?",
      answer:
        "Yes, our site is safe to use. We take security seriously and implement various measures to protect your data and privacy.",
    },
    {
      question: "How to create an article?",
      answer:
        "You need to click on the + button on the first section in the home page. After that you can write your article and add links to it. You can also add images to it.",
    },
    {
      question: "How add link in an article?",
      answer:
        "You need to write your article and to create the link you need to put it in this format: < your link >.",
    },


  ];

  return (
    <>
      <Header />
      <div className="pt-24 min-h-screen bg-gradient-to-br from-blue-600 to-blue-200 p-5">
        <h1 className="text-white text-4xl font-bold mb-10 text-center mt-10">
          -- Frequently Asked Questions --
        </h1>
        {questions.map((item, index) => (
          <div
            key={index}
            className="bg-white max-w-3xl mx-auto mb-6 p-5 rounded-2xl shadow-lg hover:scale-[1.02] transition-transform"
          >
            <div
              className="question text-lg font-bold cursor-pointer bg-blue-100 border border-blue-900 px-4 py-3 rounded-md hover:bg-blue-300 hover:text-blue-900 transition-colors text-left relative"
              onClick={() => toggleAnswer(index)}
            >
              {activeIndex === index ? '⮟' : '⮞'}{' '}
              {item.question.replace('*', '')}
              {item.question.includes('*') && (
                <span className="ml-1 relative group text-blue-900 font-bold">
                  *
                  <span className="absolute z-10 hidden group-hover:block bg-gray-800 text-white text-sm px-2 py-1 rounded-md -top-10 left-1/2 -translate-x-1/2 whitespace-nowrap shadow-lg">
                    You need to be an administrator to do this
                  </span>
                </span>
              )}
            </div>
            {activeIndex === index && (
              <div className="answer bg-blue-50 mt-4 p-4 rounded-md border-l-4 border-blue-900 text-gray-700 text-base">
                {item.answer}
              </div>
            )}
          </div>
        ))}
      </div>
      <Footer />
    </>
  );
}

export default FAQ;