// using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

using ExtensionMethods;

using edu.stanford.nlp.parser.lexparser;
using edu.stanford.nlp.ling;
using edu.stanford.nlp.objectbank;
using edu.stanford.nlp.process;
using edu.stanford.nlp.trees;
using edu.stanford.nlp.util;

using LAIR.ResourceAPIs.WordNet;
using LAIR.Collections.Generic;

using opennlp.tools.chunker;
using opennlp.tools.sentdetect;
//using opennlp.tools.tokenize;
using opennlp.tools.namefind;
using opennlp.tools.parser;
using opennlp.tools.postag;

using PorterStemmerAlgorithm;

namespace Rhetorica
{
  public sealed class Repository
  {
    // Member variables (fields, constants):
    private static string _assemblyFullName = System.Reflection.Assembly.GetExecutingAssembly().FullName;
    private static string _assemblyName = string.Empty;
    private static string _currentAssemblyDirectoryPath
      = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + Path.DirectorySeparatorChar;
    private static string _dsc = Path.DirectorySeparatorChar.ToString();

    private static string _rootDrive = string.Empty;
    private static string _nlpFolder = string.Empty;
    private static string _absTextPath = string.Empty;
    
    private static string _openNlpModelsFolder = string.Empty;
    private static string _openNlpModelsPath = string.Empty;

    private static string _wordNetFolder = string.Empty;
    private static string _wordNetPath = string.Empty;

    private static string _grammarFolder = string.Empty;
    private static string _grammarPath = string.Empty;

    private static string _dataFolder = string.Empty;
    private static string _nlpTextsPath = string.Empty;

    private static string _localTextPath = string.Empty; // For development use

    private static WordNetEngine _wordNetEngine; // WordNet engine

    private static SentenceModel _sentenceModel;
    private static SentenceDetectorME _sentenceDetector; // OpenNLP sentence detector

    private static opennlp.tools.tokenize.TokenizerModel _tokenizerModel;
    private static opennlp.tools.tokenize.TokenizerME _tokenizer; // OpenNLP tokenizer

    private static TokenNameFinderModel _tokenNameFinderModel;
    private static NameFinderME _nameFinder; // OpenNLP name finder

    private static POSModel _posModel;
    private static POSTaggerME _tagger; // OpenNLP POS tagger

    private static ChunkerModel _chunkerModel;
    private static ChunkerME _chunker; // OpenNLP chunker

    private static ParserModel _parserModel;
    private static Parser _parser; // OpenNLP parser
    private static bool _loadParser = false; // 'true' to load OpenNLP parser upon first reference

    private static LexicalizedParser _stanfordParser; // Stanford parser

    private static PorterStemmer _porterStemmer; // Porter stemmer

    // Singleton instance variable:
    // See: http://csharpindepth.com/Articles/General/Singleton.aspx
    // private static readonly Lazy<Repository> lazy = new Lazy<Repository>(() => new Repository()); // Doesn't work?
    private static readonly Repository instance = new Repository();

    // Constructors and finalizers:
    private Repository()
    {
      _assemblyName = Regex.Match(_assemblyFullName, "^(.*?),.*$").Result("$1");

      _rootDrive ="../../../../";
      _nlpFolder = ("rhetorica/nlp/").Replace(@"\", Dsc);

      _openNlpModelsFolder = ("OpenNLP/models/").Replace(@"\", Dsc);
      _openNlpModelsPath = RootDrive + _nlpFolder + _openNlpModelsFolder;

      _wordNetFolder = ("WordNet_3/").Replace(@"\", Dsc);
      _wordNetPath = RootDrive + _nlpFolder + _wordNetFolder;

      _grammarFolder = ("StanfordParser/grammar/").Replace(@"\", Dsc);
      _grammarPath = RootDrive + _nlpFolder + _grammarFolder;

      _dataFolder = ("data/").Replace(@"\", Dsc);
      _nlpTextsPath = RootDrive + _dataFolder;

      string[] localTextDirectoryParts = {
        CurrentAssemblyDirectoryPath,
        "..", "..", "..", "data"
        //"..", "..", "text"
      };
      _localTextPath = Path.Combine(localTextDirectoryParts) + "/"; // For development use

      // WordNet engine:
      Console.Write("Loading WordNet engine.... ");
      _wordNetEngine = new WordNetEngine(WordNetPath, true);
      Console.WriteLine("Done.");

      // OpenNLP sentence detector:
      Console.Write("Loading OpenNLP sentence detector.... ");
      java.io.FileInputStream modelInputStream = new java.io.FileInputStream(OpenNlpModelsPath + "en-sent.bin");
      _sentenceModel = new SentenceModel(modelInputStream);
      modelInputStream.close();
      _sentenceDetector = new SentenceDetectorME(_sentenceModel);
      Console.WriteLine("Done.");

      // OpenNLP tokenizer:
      Console.Write("Loading OpenNLP tokenizer.... ");
      modelInputStream = new java.io.FileInputStream(OpenNlpModelsPath + "en-token.bin");
      _tokenizerModel = new opennlp.tools.tokenize.TokenizerModel(modelInputStream);
      modelInputStream.close();
      _tokenizer = new opennlp.tools.tokenize.TokenizerME(_tokenizerModel);
      Console.WriteLine("Done.");

      // OpenNLP name finder:
      Console.Write("Loading OpenNLP name finder.... ");
      modelInputStream = new java.io.FileInputStream(OpenNlpModelsPath + "en-ner-person.bin");
      _tokenNameFinderModel = new TokenNameFinderModel(modelInputStream);
      modelInputStream.close();
      _nameFinder = new NameFinderME(_tokenNameFinderModel);
      Console.WriteLine("Done.");

      // OpenNLP POS tagger:
      Console.Write("Loading OpenNLP POS tagger.... ");
      modelInputStream = new java.io.FileInputStream(OpenNlpModelsPath + "en-pos-maxent.bin");
      _posModel = new POSModel(modelInputStream);
      modelInputStream.close();
      _tagger = new POSTaggerME(_posModel);
      Console.WriteLine("Done.");

      // OpenNLP chunker:
      Console.Write("Loading OpenNLP chunker.... ");
      modelInputStream = new java.io.FileInputStream(OpenNlpModelsPath + "en-chunker.bin");
      _chunkerModel = new ChunkerModel(modelInputStream);
      modelInputStream.close();
      _chunker = new ChunkerME(_chunkerModel);
      Console.WriteLine("Done.");

      // OpenNLP parser:
      if (_loadParser) {
        Console.Write("Loading OpenNLP parser.... ");
        modelInputStream = new java.io.FileInputStream(OpenNlpModelsPath + "en-parser-chunking.bin");
        _parserModel = new ParserModel(modelInputStream);
        modelInputStream.close();
        _parser = ParserFactory.create(_parserModel);
        Console.WriteLine("Done.");
      }

      // Stanford parser:
      //_stanfordParser = new LexicalizedParser(GrammarPath + "englishPCFG.ser.gz"); // Obsolete method
      _stanfordParser = LexicalizedParser.loadModel(GrammarPath + "englishPCFG.ser.gz");

      // Porter stemmer:
      _porterStemmer = new PorterStemmer();
    }

    // Enums, Structs, and Classes:

    // Properties:
    public static Repository Instance
    {
      //get { return lazy.Value; }
      get { return instance; }
    }

    public static string AssemblyName
    {
      get { return _assemblyName; }
    }

    public static string AssemblyFullName
    {
      get { return _assemblyFullName; }
    }

    public static string CurrentAssemblyDirectoryPath
    {
      get { return _currentAssemblyDirectoryPath; }
    }

    public static string Dsc
    {
      get { return _dsc; }
    }

    public static string RootDrive
    {
      get { return _rootDrive; }
      set { 
      
        _rootDrive = value; 
      
        _openNlpModelsPath = RootDrive + _nlpFolder + _openNlpModelsFolder;
        
        _wordNetPath = RootDrive + _nlpFolder + _wordNetFolder;
        
        _grammarPath = RootDrive + _nlpFolder + _grammarFolder;
        
        _nlpTextsPath = RootDrive + _dataFolder;
      }
    }
    
    public static string AbsTextPath
    {
        get { return _absTextPath; }
        set {
            _absTextPath = value;
        }
     }
    
    public static string OpenNlpModelsPath
    {
      get { return _openNlpModelsPath; }
    }

    public static string WordNetPath
    {
      get { return _wordNetPath; }
    }

    public static string GrammarPath
    {
      get { return _grammarPath; }
    }

    public static string NlpTextsPath
    {
      get { return _nlpTextsPath; }
    }
        
    public static string LocalTextPath
    {
      get { return _localTextPath; }
    }

    public static WordNetEngine WordNet
    {
      get { return _wordNetEngine; }
    }

    public static SentenceDetectorME SentenceDetector
    {
      get { return _sentenceDetector; }
    }

    public static opennlp.tools.tokenize.TokenizerME Tokenizer
    {
      get { return _tokenizer; }
    }

    public static NameFinderME NameFinder
    {
      get { return _nameFinder; }
    }

    public static POSTaggerME Tagger
    {
      get { return _tagger; }
    }

    public static ChunkerME Chunker
    {
      get { return _chunker; }
    }

    public static Parser OpenNlpParser
    {
      get { return _parser; }
    }

    public static LexicalizedParser StanfordParser
    {
      get { return _stanfordParser; }
    }

    public static PorterStemmer PorterStemmer
    {
      get { return _porterStemmer; }
    }
  }
}
