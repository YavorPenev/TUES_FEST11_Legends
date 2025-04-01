import React, { useState } from 'react';

function FAQ() {
  const [activeIndex, setActiveIndex] = useState(null);

  const toggleAnswer = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  const questions = [
    {
      question: "How to create an account in the site?",
      answer:
        "To create an account on the website, click on the 'Sign Up' or 'Create Account' button, usually located at the top-right corner of the homepage. You will be prompted to provide your username, email address, and a password. Some websites may also require you to confirm your email address by clicking a verification link sent to your inbox. Once your account is created, you can log in and start contributing to the website.",
    },
    {
      question: "How do I log in to my account?",
      answer:
        "To log in, click on the 'Login' button at the top-right corner of the homepage. Enter your registered email address and password, then click 'Submit'. If you’ve forgotten your password, use the 'Forgot Password' option to reset it.",
    },
    {
      question: "How to use the loan calculator?",
      answer:
        "To use the loan calculator, navigate to the 'Loan Calculator' page from the menu. Enter the loan amount, interest rate, and repayment period. The calculator will display your monthly payment and total repayment amount.",
    },
    {
      question: "How to change the theme?",
      answer:
        "To change the theme, click on the 'Theme' button located in the top-right corner of the page. You can choose between light and dark modes or customize the theme colors according to your preference.",
    },
    {
      question: "How to use the currency exchange calculator?",
      answer:
        "To use the currency exchange calculator, go to the 'Currency Exchange' page. Select the currencies you want to convert between, enter the amount, and the calculator will display the converted value based on the latest exchange rates.",
    },
    {
      question: "How can I track changes in an article I'm interested in?",
      answer:
        "To track changes in an article, use the 'Watch' feature available on most platforms. By clicking the 'Watch' button, you’ll receive notifications whenever someone makes an edit to the article.",
    },
    {
      question: "How do I remove or modify information I’ve added?",
      answer:
        "To remove or modify information, go to the page and click the 'Edit' button. Make the necessary changes, preview the article, and save your edits.",
    },
    {
      question: "What should I do if an article has been vandalized?",
      answer:
        "If an article has been vandalized, use the 'Undo' feature to revert the changes. If you cannot undo the changes, report the issue to the site administrators or moderators.",
    },
    {
      question: "How can I suggest a new topic or article to be added?",
      answer:
        "To suggest a new topic, search the site to ensure the subject doesn’t already exist. If it’s new, use the 'Create Article' option to start a new page. Ensure the topic is notable and has reliable sources.",
    },
    {
      question: "How can I leave a review about the site?",
      answer:
        "To leave a review, navigate to the 'Feedback' or 'Contact Us' section of the website. Provide your feedback in the form provided, or send an email to the support team.",
    },
  ];

  return (
    <div className="p-5 ">
      <h1 className="text-blue-500 text-4xl font-bold mb-10 text-center">
        -- Frequently Asked Questions --
      </h1>
      {questions.map((item, index) => (
        <div
          key={index}
          className="section mb-5 bg-blue-200 p-4 rounded-lg shadow-lg"
          style={{  margin: '3%'}} 
      
        >
          <div
            className="question font-bold text-lg cursor-pointer text-left"
            onClick={() => toggleAnswer(index)}
          >
            ⮞ {item.question}
          </div>
          {activeIndex === index && (
            <div className="answer mt-3 text-gray-700 text-left">
              {item.answer}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default FAQ;