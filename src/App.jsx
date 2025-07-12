import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Login, Home } from './pages'

const App = () => {
  const routes =(
    <Routes>
      <Route path="/home" element={<Home/>} />
      <Route path="/login" element={<Login/>} />
    </Routes>
  )
  return (
    <>
      {routes}
    </>
  )
}

export default App
