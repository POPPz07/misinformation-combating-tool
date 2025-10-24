import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router-dom";
import axios from "axios";
import { AlertCircle, CheckCircle, XCircle, Upload, FileText, History, Info, User, LogOut, Shield, Search, Camera, FileImage, ExternalLink } from "lucide-react";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Textarea } from "./components/ui/textarea";
import { Progress } from "./components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { toast } from "sonner";
import { Toaster } from "./components/ui/sonner";
import "./App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = BACKEND_URL;
// Auth Context
const AuthContext = React.createContext();

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const response = await axios.get(`${API}/auth/me`, { withCredentials: true });
      setUser(response.data);
    } catch (error) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = () => {
    const redirectUrl = encodeURIComponent(window.location.origin + '/profile');
    window.location.href = `https://auth.emergentagent.com/?redirect=${redirectUrl}`;
  };

  const logout = async () => {
    try {
      await axios.post(`${API}/auth/logout`, {}, { withCredentials: true });
      setUser(null);
      toast.success("Logged out successfully!");
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <AuthContext.Provider value={{ user, setUser, login, logout, loading, checkAuth }}>
      {children}
    </AuthContext.Provider>
  );
};

const useAuth = () => React.useContext(AuthContext);

// Components
const Navbar = () => {
  const { user, login, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <nav className="bg-white/90 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div 
            className="flex items-center space-x-2 cursor-pointer" 
            onClick={() => navigate('/')}
          >
            <Shield className="h-8 w-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-900">TruthLens</span>
          </div>
          
          <div className="flex items-center space-x-4">
            <Button 
              variant="ghost" 
              onClick={() => navigate('/workspace')}
              className="text-gray-700 hover:text-blue-600"
            >
              <Search className="h-4 w-4 mr-2" />
              Analyze
            </Button>
            <Button 
              variant="ghost" 
              onClick={() => navigate('/history')}
              className="text-gray-700 hover:text-blue-600"
            >
              <History className="h-4 w-4 mr-2" />
              History
            </Button>
            <Button 
              variant="ghost" 
              onClick={() => navigate('/about')}
              className="text-gray-700 hover:text-blue-600"
            >
              <Info className="h-4 w-4 mr-2" />
              About
            </Button>
            
            {user ? (
              <div className="flex items-center space-x-2">
                <Button 
                  variant="ghost" 
                  onClick={() => navigate('/profile')}
                  className="text-gray-700 hover:text-blue-600"
                >
                  <User className="h-4 w-4 mr-2" />
                  {user.name}
                </Button>
                <Button 
                  variant="ghost" 
                  onClick={logout}
                  size="sm"
                >
                  <LogOut className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <Button onClick={login} className="bg-blue-600 hover:bg-blue-700">
                Sign In
              </Button>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Hero Section */}
      <div className="pt-20 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <div className="mb-8">
            <Shield className="h-20 w-20 text-blue-600 mx-auto mb-6" />
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
              Truth<span className="text-blue-600">Lens</span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Detect misinformation. Learn to spot the truth.
            </p>
          </div>
          
          <Button 
            onClick={() => navigate('/workspace')}
            size="lg"
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 text-lg rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
          >
            Try Now
          </Button>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16 px-4 sm:px-6 lg:px-8 bg-white/60 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
            Powerful Features
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="text-center border-0 shadow-lg hover:shadow-xl transition-all duration-300">
              <CardHeader>
                <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                <CardTitle className="text-xl">Fact-Checking</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">
                  Advanced AI analysis to verify claims and identify potential misinformation in text and images.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center border-0 shadow-lg hover:shadow-xl transition-all duration-300">
              <CardHeader>
                <AlertCircle className="h-12 w-12 text-orange-500 mx-auto mb-4" />
                <CardTitle className="text-xl">Scam Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">
                  Identify suspicious patterns, emotional manipulation, and common misinformation tactics.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center border-0 shadow-lg hover:shadow-xl transition-all duration-300">
              <CardHeader>
                <Info className="h-12 w-12 text-blue-500 mx-auto mb-4" />
                <CardTitle className="text-xl">Education</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">
                  Learn practical tips and strategies to identify misinformation and verify sources independently.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

const WorkspacePage = () => {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [analysisType, setAnalysisType] = useState('text');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeContent = async () => {
    if (!input.trim() && !imageFile) {
      toast.error("Please enter text or upload an image to analyze");
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      let requestData = {
        content_type: analysisType
      };

      if (analysisType === 'text') {
        requestData.text = input;
      } else if (analysisType === 'image' && imageFile) {
        // Convert image to base64
        const reader = new FileReader();
        const base64Promise = new Promise((resolve, reject) => {
          reader.onload = () => {
            const base64 = reader.result.split(',')[1]; // Remove data:image/jpeg;base64, prefix
            resolve(base64);
          };
          reader.onerror = reject;
        });
        reader.readAsDataURL(imageFile);
        const base64Image = await base64Promise;
        requestData.image_base64 = base64Image;
        requestData.text = input || "Analyze this image for misinformation";
      }

      const response = await axios.post(`${API}/analyze`, requestData, {
        withCredentials: true
      });

      setResult(response.data);
      toast.success("Analysis completed!");
    } catch (error) {
      console.error('Analysis error:', error);
      toast.error("Failed to analyze content. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 70) return "text-green-600";
    if (score >= 30) return "text-orange-600";
    return "text-red-600";
  };

  const getScoreIcon = (score) => {
    if (score >= 70) return <CheckCircle className="h-5 w-5 text-green-600" />;
    if (score >= 30) return <AlertCircle className="h-5 w-5 text-orange-600" />;
    return <XCircle className="h-5 w-5 text-red-600" />;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Panel - Input */}
          <div className="lg:col-span-1">
            <Card className="sticky top-24">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <FileText className="h-5 w-5 mr-2" />
                  Content Analysis
                </CardTitle>
                <CardDescription>
                  Enter text or upload an image to analyze for misinformation
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Tabs value={analysisType} onValueChange={setAnalysisType}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="text">
                      <FileText className="h-4 w-4 mr-2" />
                      Text
                    </TabsTrigger>
                    <TabsTrigger value="image">
                      <Camera className="h-4 w-4 mr-2" />
                      Image
                    </TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="text" className="space-y-4">
                    <Textarea 
                      placeholder="Paste your text here (news article, social media post, etc.)"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      rows={8}
                      className="resize-none"
                    />
                  </TabsContent>
                  
                  <TabsContent value="image" className="space-y-4">
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                      <input
                        type="file"
                        id="image-upload"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="hidden"
                      />
                      <label htmlFor="image-upload" className="cursor-pointer">
                        {imagePreview ? (
                          <img src={imagePreview} alt="Preview" className="max-w-full h-32 object-contain mx-auto mb-2" />
                        ) : (
                          <FileImage className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                        )}
                        <p className="text-sm text-gray-600">
                          {imagePreview ? "Click to change image" : "Click to upload image"}
                        </p>
                      </label>
                    </div>
                    <Textarea 
                      placeholder="Optional: Add context about the image"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      rows={3}
                      className="resize-none"
                    />
                  </TabsContent>
                </Tabs>

                <Button 
                  onClick={analyzeContent}
                  disabled={isLoading || (!input.trim() && !imageFile)}
                  className="w-full bg-blue-600 hover:bg-blue-700"
                  size="lg"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Search className="h-4 w-4 mr-2" />
                      Analyze Content
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Center Panel - Results */}
          <div className="lg:col-span-2">
            {result ? (
              <div className="space-y-6">
                {/* Credibility Score */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <span>Analysis Results</span>
                      <Badge variant={result.credibility_score >= 70 ? "default" : result.credibility_score >= 30 ? "secondary" : "destructive"}>
                        {result.verdict}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">Credibility Score</span>
                          <div className="flex items-center space-x-2">
                            {getScoreIcon(result.credibility_score)}
                            <span className={`text-lg font-bold ${getScoreColor(result.credibility_score)}`}>
                              {result.credibility_score}/100
                            </span>
                          </div>
                        </div>
                        <Progress value={result.credibility_score} className="h-3" />
                      </div>
                      
                      <div>
                        <span className="text-sm font-medium">Confidence Level: </span>
                        <span className="text-sm text-gray-600">
                          {Math.round(result.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Reasoning */}
                <Card>
                  <CardHeader>
                    <CardTitle>Analysis Reasoning</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-700 leading-relaxed">{result.reasoning}</p>
                  </CardContent>
                </Card>

                {/* Education Tips */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Info className="h-5 w-5 mr-2" />
                      Education Tips
                    </CardTitle>
                    <CardDescription>
                      Learn how to identify similar misinformation
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {result.education_tips.map((tip, index) => (
                        <li key={index} className="flex items-start">
                          <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{tip}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card className="h-96 flex items-center justify-center">
                <CardContent className="text-center">
                  <Search className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to Analyze</h3>
                  <p className="text-gray-500">
                    Enter content in the left panel and click "Analyze Content" to get started
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        {/* Quick Resources Panel */}
        <div className="mt-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Info className="h-5 w-5 mr-2" />
                Quick Resources
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                <a href="https://www.altnews.in/" target="_blank" rel="noopener noreferrer" 
                   className="text-blue-600 hover:text-blue-800 text-sm">AltNews</a>
                <a href="https://pib.gov.in/indexd.aspx" target="_blank" rel="noopener noreferrer" 
                   className="text-blue-600 hover:text-blue-800 text-sm">PIB Fact Check</a>
                <a href="https://www.who.int/" target="_blank" rel="noopener noreferrer" 
                   className="text-blue-600 hover:text-blue-800 text-sm">WHO</a>
                <a href="https://www.snopes.com/" target="_blank" rel="noopener noreferrer" 
                   className="text-blue-600 hover:text-blue-800 text-sm">Snopes</a>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

const HistoryPage = () => {
  const { user } = useAuth();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
  }, [user]);

  const fetchHistory = async () => {
    try {
      if (user) {
        const response = await axios.get(`${API}/history`, { withCredentials: true });
        setHistory(response.data.analyses);
      } else {
        // Get public history for non-authenticated users
        const response = await axios.get(`${API}/public-history`);
        setHistory(response.data.analyses || []);
      }
    } catch (error) {
      console.error('Error fetching history:', error);
      toast.error("Failed to load history");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-300 rounded w-1/4 mb-8"></div>
            <div className="space-y-4">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-32 bg-gray-300 rounded"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Analysis History</h1>
          <p className="text-gray-600">
            {user ? "Your previous analyses" : "Recent public analyses"}
          </p>
        </div>

        {history.length === 0 ? (
          <Card>
            <CardContent className="text-center py-12">
              <History className="h-16 w-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No History Yet</h3>
              <p className="text-gray-500 mb-4">
                {user ? "You haven't analyzed any content yet." : "No public analyses available."}
              </p>
              <Button onClick={() => window.location.href = '/workspace'}>
                Start Analyzing
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {history.map((item) => (
              <Card key={item.id} className="hover:shadow-lg transition-shadow duration-200">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <Badge variant={item.credibility_score >= 70 ? "default" : item.credibility_score >= 30 ? "secondary" : "destructive"}>
                          {item.verdict}
                        </Badge>
                        <span className="text-sm text-gray-500">
                          {new Date(item.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                      <p className="text-gray-700 mb-2">{item.content}</p>
                      <div className="flex items-center space-x-4">
                        <span className="text-sm text-gray-600">
                          Score: <span className="font-medium">{item.credibility_score}/100</span>
                        </span>
                        <span className="text-sm text-gray-600">
                          Confidence: <span className="font-medium">{Math.round(item.confidence * 100)}%</span>
                        </span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const AboutPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <Shield className="h-20 w-20 text-blue-600 mx-auto mb-6" />
          <h1 className="text-4xl font-bold text-gray-900 mb-4">About TruthLens</h1>
          <p className="text-xl text-gray-600">
            Empowering people to identify and combat misinformation
          </p>
        </div>

        <div className="space-y-8">
          <Card>
            <CardHeader>
              <CardTitle>Our Mission</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-700 leading-relaxed">
                TruthLens is dedicated to fighting misinformation by providing advanced AI-powered analysis tools
                and educational resources. We believe that everyone deserves access to accurate information and
                the skills to identify misleading content.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>How It Works</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start">
                  <div className="bg-blue-100 rounded-full p-2 mr-4">
                    <span className="text-blue-600 font-bold text-sm">1</span>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">Submit Content</h4>
                    <p className="text-gray-600">Upload text or images that you want to verify</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="bg-blue-100 rounded-full p-2 mr-4">
                    <span className="text-blue-600 font-bold text-sm">2</span>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">AI Analysis</h4>
                    <p className="text-gray-600">Our advanced AI analyzes the content for misinformation indicators</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="bg-blue-100 rounded-full p-2 mr-4">
                    <span className="text-blue-600 font-bold text-sm">3</span>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">Get Results</h4>
                    <p className="text-gray-600">Receive a credibility score, verdict, and educational tips</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Impact</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-700 leading-relaxed">
                By providing accessible misinformation detection tools, we aim to create a more informed society
                where people can make decisions based on accurate information. Every analysis helps build a
                database of knowledge that benefits the entire community.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

const ProfilePage = () => {
  const { user, setUser, checkAuth } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    // Handle OAuth redirect
    const handleOAuthRedirect = async () => {
      const hash = location.hash;
      if (hash.includes('session_id=')) {
        const sessionId = hash.split('session_id=')[1];
        
        try {
          const response = await axios.post(`${API}/auth/session`, 
            { session_id: sessionId },
            { withCredentials: true }
          );
          
          setUser(response.data.user);
          toast.success("Successfully logged in!");
          navigate('/profile', { replace: true });
        } catch (error) {
          console.error('Session creation error:', error);
          toast.error("Login failed. Please try again.");
          navigate('/', { replace: true });
        }
      }
    };

    if (location.hash.includes('session_id=')) {
      handleOAuthRedirect();
    } else if (!user) {
      // Refresh auth state
      checkAuth();
    }
  }, [location.hash, user, setUser, navigate, checkAuth]);

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-8">
        <div className="max-w-md mx-auto px-4 sm:px-6 lg:px-8">
          <Card>
            <CardContent className="text-center py-12">
              <User className="h-16 w-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Authentication Required</h3>
              <p className="text-gray-500 mb-4">Please sign in to view your profile</p>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-8">
      <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <User className="h-5 w-5 mr-2" />
              Profile
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-4 mb-6">
              {user.picture && (
                <img 
                  src={user.picture} 
                  alt={user.name}
                  className="h-16 w-16 rounded-full"
                />
              )}
              <div>
                <h2 className="text-xl font-bold text-gray-900">{user.name}</h2>
                <p className="text-gray-600">{user.email}</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <Button 
                onClick={() => navigate('/history')}
                variant="outline"
                className="w-full justify-start"
              >
                <History className="h-4 w-4 mr-2" />
                View Analysis History
              </Button>
              
              <Button 
                onClick={() => navigate('/workspace')}
                className="w-full justify-start bg-blue-600 hover:bg-blue-700"
              >
                <Search className="h-4 w-4 mr-2" />
                Analyze New Content
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Main App
function App() {
  return (
    <AuthProvider>
      <div className="App">
        <BrowserRouter>
          <Navbar />
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/workspace" element={<WorkspacePage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/profile" element={<ProfilePage />} />
          </Routes>
          <Toaster position="top-right" />
        </BrowserRouter>
      </div>
    </AuthProvider>
  );
}

export default App;