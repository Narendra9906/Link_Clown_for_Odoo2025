import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { profile_logo } from '../assets';
import axios from 'axios';

const Login = () => {
    const [isToggled, setIsToggled] = useState(false); // Restored useState for toggle
    const [identifier, setIdentifier] = useState(''); // Combined email and username
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    // Restored handleToggle function
    const handleToggle = () => {
        setIsToggled(!isToggled);
    };

    const handleLogin = async (e) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            const response = await axios.post('http://localhost:5000/api/users/login', {
                identifier,
                password,
            });

            // Store the token in localStorage
            if (response.data.token) {
                localStorage.setItem('token', response.data.token);
                localStorage.setItem('userData', JSON.stringify(response.data));

                // If remember me is toggled, set a longer expiration
                if (isToggled) {
                    // The token itself has an expiration, this is just to track the user preference
                    localStorage.setItem('rememberMe', 'true');
                }

                navigate('/home');
            }
        } catch (err) {
            setError(err.response?.data?.message || 'Login failed. Please check your credentials.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className='relative bg-no-repeat bg-cover'>
            <div className='absolute bg-[#DEDDEA] inset-0 flex justify-center items-center h-screen'>
                <div className='flex w-100 h-120 bg-white z-10 rounded-xl shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1),_6px_0_6px_-3px_rgba(0,0,0,0.1),_-6px_0_6px_-3px_rgba(0,0,0,0.1)]'>
                    <div className='w-full flex flex-col px-4 py-8 relative'>


                        <div className=' mr-6'>
                            <h1 className='text-2xl font-bold'>Login to ReWear clothing exchange !</h1>

                            <img
                                src={profile_logo}
                                alt="Profile Logo"
                                className='rounded-full w-16 h-16 mb-4 mt-2 mx-auto'
                                />

                           

                            {error && (
                                <div className='mb-2 p-1 bg-red-100 border border-red-400 text-red-700 rounded text-xs'>
                                    {error}
                                </div>
                            )}

                            <form onSubmit={handleLogin} className='flex gap-1 flex-col'>
                                <label className='text-gray-700 text-medium font-medium'>Email or Username</label>
                                <div className='relative mb-1'>
                                    <input
                                        type="email"
                                        className='w-full py-1.5 px-2 border border-gray-300 rounded-md text-sm'
                                        style={{ height: '32px' }}
                                        value={identifier}
                                        onChange={(e) => setIdentifier(e.target.value)}
                                        placeholder="Enter your email or username"
                                        required
                                    />
                                    <div className='absolute right-2 top-1.5 text-gray-500'>
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            className="h-5 w-5"
                                            fill="none"
                                            viewBox="0 0 24 24"
                                            stroke="currentColor"
                                        >
                                            <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                strokeWidth={2}
                                                d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                                            />
                                        </svg>
                                    </div>
                                </div>

                                <label className='text-gray-700 text-medium font-medium'>Password</label>
                                <div className='relative mb-2'>
                                    <input
                                        type="password"
                                        className='w-full py-1.5 px-2 border border-gray-300 rounded-md text-sm h-6'
                                        style={{ height: '32px' }}
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        required
                                    />
                                    <div className='absolute right-2 top-1.5 text-gray-500'>
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            className="h-5 w-5"
                                            fill="none"
                                            viewBox="0 0 24 24"
                                            stroke="currentColor"
                                        >
                                            <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                strokeWidth={2}
                                                d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                                            />
                                        </svg>
                                    </div>
                                </div>

                                <div className='flex justify-between items-center mb-2'>
                                    <div className='flex items-center'>
                                        <label className="relative inline-flex items-center cursor-pointer">
                                            <input
                                                type="checkbox"
                                                className="sr-only peer"
                                                checked={isToggled}
                                                onChange={handleToggle} // Updated to use handleToggle
                                            />
                                            <div className="w-9 h-5 bg-gray-300 rounded-full peer peer-checked:bg-[#482D64] peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all"></div>
                                        </label>
                                        <span className='text-xs text-gray-700 ml-2 mr-2'>Remember Me</span>
                                    </div>
                                    <span className='text-xs text-gray-700 hover:text-blue-500 cursor-pointer'>Forgot Password ?</span>
                                </div>

                                <button
                                    type="submit"
                                    disabled={isLoading}
                                    className='bg-[#6c4298] hover:bg-[#482D64] cursor-pointer text-white font-medium text-medium py-1.5 rounded-full mt-1 mb-2'
                                >
                                    {isLoading ? 'Logging in...' : 'Login'}
                                </button>
                            </form>
                        </div>

                        <div className='flex items-center justify-center py-1'>
                            <div className='flex-1 border-t border-gray-300'></div>
                            <span className='mx-2 text-xs text-gray-500'>or</span>
                            <div className='flex-1 border-t border-gray-300'></div>
                        </div>

                        {/* Sign up button */}
                        <div className='flex items-center justify-center mt-2'>
                            <p className='text-xs text-gray-700'>Don't have an account ?</p>
                            <button
                                onClick={() => navigate('/signup')}
                                className='text-blue-400 hover:text-blue-600 cursor-pointer ml-2 text-xs'
                            >
                                Sign up
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Login;