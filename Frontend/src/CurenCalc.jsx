import { useEffect, useState } from 'react';
import axios from 'axios';
import Header from './assets/header';
import Footer from './assets/footer';
import Calculator from './assets/SimpCalc';
import Notes from './assets/notes';


function CurrencyCalc() {

  const [searchTerm, setSearchTerm] = useState('');
  const [sortCondition, setSortCondition] = useState('default');
  const [rates, setRates] = useState(null);
  const [error, setError] = useState(null);
  const [amount, setAmount] = useState(1);
  const [fromCurrency, setFromCurrency] = useState('USD');
  const [toCurrency, setToCurrency] = useState('EUR')
  const [result, setResult] = useState(null);

  const currencyRegions = {
    USD: 'North America',
    CAD: 'North America',
    MXN: 'North America',

    EUR: 'Europe',
    GBP: 'Europe',
    CHF: 'Europe',
    NOK: 'Europe',
    SEK: 'Europe',
    DKK: 'Europe',
    PLN: 'Europe',
    HUF: 'Europe',
    BGN: 'Europe',
    CZK: 'Europe',

    JPY: 'Asia',
    CNY: 'Asia',
    INR: 'Asia',
    KRW: 'Asia',
    SGD: 'Asia',
    HKD: 'Asia',

    AUD: 'Oceania',
    NZD: 'Oceania',

    BRL: 'South America',
    ARS: 'South America',
    CLP: 'South America',

    ZAR: 'Africa',
    EGP: 'Africa',
    NGN: 'Africa',

    TRY: 'Other',
  };

  const [regionFilter, setRegionFilter] = useState('All');

  const getSortedRates = () => {
    let entries = Object.entries(rates);

    // Apply region filter
    if (regionFilter !== 'All') {
      entries = entries.filter(([currency]) => currencyRegions[currency] === regionFilter);
    }

    if (regionFilter !== 'All') {
      entries = entries.filter(([currency]) => currencyRegions[currency] === regionFilter);
    }

    // Apply search filter
    if (searchTerm) {
      entries = entries.filter(([currency]) => currency.includes(searchTerm));
    }

    // Sorting logic
    switch (sortCondition) {
      case 'highest':
        return entries.sort((a, b) => b[1] - a[1]);
      case 'lowest':
        return entries.sort((a, b) => a[1] - b[1]);
      case 'alphabetical':
        return entries.sort((a, b) => a[0].localeCompare(b[0]));
      case 'reverse':
        return entries.sort((a, b) => b[0].localeCompare(a[0]));
      case 'default':
      default:
        return entries.sort((a, b) => {
          if (a[0] === 'USD') return -1;
          if (b[0] === 'USD') return 1;
          return a[0].localeCompare(b[0]);
        });
    }
  };

  useEffect(() => {
    const fetchRates = async () => {
      try {
        const response = await axios.get('https://api.exchangerate-api.com/v4/latest/USD'); // API за валутни курсове
        setRates(response.data.rates);
      } catch (err) {
        setError('Error fetching exchange rates. Please try again later.');
      }
    };
    fetchRates();
  }, []);

  const handleConvert = () => {
    if (rates) {
      const fromRate = rates[fromCurrency];
      const toRate = rates[toCurrency];
      const convertedAmount = (amount / fromRate) * toRate;
      setResult(convertedAmount.toFixed(2));
    }
  };

  return (
    <>
      <Header />
      <div className="p-5 pt-24" style={{ margin: '0 10%' }}>
        <h1 className="text-3xl font-bold mb-6 text-center mt-20">Currency Calculator</h1>
        {error && <p className="text-red-500 text-center">{error}</p>}
        {rates ? (
          <div >
            <div className="flex flex-wrap items-center justify-center gap-4 mb-6">
              <label className="font-semibold">Amount:</label>
              <input
                type="number"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                className="p-2 border rounded w-32"
              />
              <label className="font-semibold">From:</label>
              <select
                value={fromCurrency}
                onChange={(e) => setFromCurrency(e.target.value)}
                className="p-2 border rounded"
              >
                {Object.keys(rates).map((currency) => (
                  <option key={currency} value={currency}>
                    {currency}
                  </option>
                ))}
              </select>
              <label className="font-semibold">To:</label>
              <select
                value={toCurrency}
                onChange={(e) => setToCurrency(e.target.value)}
                className="p-2 border rounded"
              >
                {Object.keys(rates).map((currency) => (
                  <option key={currency} value={currency}>
                    {currency}
                  </option>
                ))}
              </select>
              <button
                onClick={handleConvert}
                className="bg-blue-800 text-white px-4 py-2 rounded hover:scale-110 active:scale-90 transition-transform hover:duration-200 active:duration-50 font-bold"
              >
                Convert
              </button>
            </div>


            {result && (
              <div className="text-center mb-8">
                <h2 className="text-2xl font-semibold">
                  {amount} {fromCurrency} = {result} {toCurrency}
                </h2>
              </div>
            )}

            <div className="bg-blue-100 p-4 rounded-lg shadow mb-6">
              <p className="text-lg text-gray-700 text-center">
                Easily convert currencies with real-time exchange rates. Simply enter the amount, select the currencies, and click <b>"Convert"</b> to get the latest conversion. Stay updated with accurate rates and make informed financial decisions effortlessly!
              </p>
            </div>

            <div className='flex flex-row items-center justify-center gap-6 mb-4 mt-8'>
              <input
                type="text"
                placeholder="Search currency (e.g. USD)"
                className="p-2 border rounded w-full max-w-sm mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value.toUpperCase())}
              />

              <div className="flex items-center justify-end mb-4 gap-4">
                {/* Region Filter */}
                <div className="flex items-center">
                  <label className="mr-2 font-semibold">Filter by Region:</label>
                  <select
                    value={regionFilter}
                    onChange={(e) => setRegionFilter(e.target.value)}
                    className="p-2 border rounded"
                  >
                    <option value="All">All</option>
                    <option value="North America">North America</option>
                    <option value="Europe">Europe</option>
                    <option value="Asia">Asia</option>
                    <option value="Oceania">Oceania</option>
                    <option value="South America">South America</option>
                    <option value="Africa">Africa</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                {/* Sort Dropdown (existing) */}
                <div className="flex items-center">
                  <label className="mr-2 font-semibold">Sort by:</label>
                  <select
                    value={sortCondition}
                    onChange={(e) => setSortCondition(e.target.value)}
                    className="p-2 border rounded"
                  >
                    <option value="default">Default (USD First, A-Z)</option>
                    <option value="alphabetical">Alphabetical (A-Z)</option>
                    <option value="reverse">Reverse Alphabetical (Z–A)</option>
                    <option value="highest">Highest Rate First</option>
                    <option value="lowest">Lowest Rate First</option>
                  </select>
                </div>
              </div>
            </div>

            <h2 className="text-3xl font-bold mt-4 mb-4">Exchange Rates</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4 max-h-[600px] overflow-y-auto p-6 bg-blue-200 rounded-xl">
              {getSortedRates().map(([currency, rate]) => (
                <div key={currency} className="p-3 bg-blue-800 rounded shadow text-white hover:scale-110 hover:duration-200">
                  <p className="font-semibold">{currency}:</p>
                  <p>{rate.toFixed(2)}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p>Loading...</p>
        )}
      </div>
      <Notes />
      <Calculator />
      <Footer />
    </>
  );
}

export default CurrencyCalc;