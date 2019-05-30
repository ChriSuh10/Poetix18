// using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
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
  public class Document
  {
    // Member variables (fields, constants):
    protected string _ignore;
    protected string _punctuation;
    protected AnalyzerOptions _options;

    // Constructors and finalizers:
    protected Document() // Don't allow unparameterized instantiation
    {
    }

    public Document(string rawText, string name, string lineBreakReplacement = " ", string ignore = "", string punctuation = null, AnalyzerOptions options = AnalyzerOptions.None)
      : this()
    {
      Name = name;
      RawText = rawText;

      _ignore = ignore;
      _punctuation = punctuation;
      _options = options;

      Resolve(lineBreakReplacement);
    }

    // Properties:
    public string Name
    {
      get; protected set;
    }

    public string RawText
    {
      get; protected set;
    }

    public string Text
    {
      get { return GetCleanText(); }
    }

    public DocumentPreprocessor Preprocessor
    {
      get; set;
    }

    public string Ignore
    {
      get { return _ignore; }
    }

    public string Punctuation
    {
      get { return _punctuation; }
    }

    public AnalyzerOptions Options
    {
      get { return _options; }
    }

    public List<Sentence> Sentences
    {
      get; protected set;
    }

    // Methods:
    private void Resolve(string lineBreakReplacement = " ")
    {
      Sentences = new List<Sentence>();

      // Is there any benefit to using 'edu.stanford.nlp.process.DocumentPreprocessor' here?
      // V. http://stackoverflow.com/questions/9492707/how-can-i-split-a-text-into-sentences-using-the-stanford-parser
      java.io.Reader reader = new java.io.StringReader(RawText);
      DocumentPreprocessor dp = new DocumentPreprocessor(reader);
            dp.setSentenceDelimiter("\n"); // This adds the semicolon as a sentence delimiter.
      var sentenceList = new List<string>();
      for (java.util.Iterator i = dp.iterator(); i.hasNext(); ) {
        java.util.List sentence = (java.util.List)i.next();
        sentenceList.Add(edu.stanford.nlp.ling.Sentence.listToString(sentence)); // Overidden by 'Rhetorica.Sentence'.
      }
      var sentencesStanford = sentenceList.ToArray();
      // N.B. This does basically the same thing as 'opennlp.tools.sentdetect.SentenceDetectorME.sentDetect()', but allows changing sentence-ending punctuation.

      //string[] sentences = Repository.SentenceDetector.sentDetect(RawText);
      var sentences = sentencesStanford; // Use instead the sentences detected above by the Stanford preprocessor.
      Console.OutputEncoding = Encoding.GetEncoding("iso-8859-1");
      StreamWriter file = new StreamWriter(Repository.NlpTextsPath + "rhet/" + Path.GetFileNameWithoutExtension(Repository.AbsTextPath) + "_" + "corpus.txt");
      file.Write("id|text\n");
      for (int i = 0; i < sentences.Length; ++i) {
        Console.Write("Parsing sentence {0}/{2}.... {1}\n", i + 1,sentences[i],sentences.Length);
        file.Write("{0}|{1}\n", i, sentences[i]);
        Sentences.Add(new Sentence(sentences[i], lineBreakReplacement, _ignore, _punctuation, _options, this));
        Console.WriteLine("Done.");
      }
     file.Close();
    }

    private string GetCleanText()
    {
      string cleanText = string.Empty;
      foreach (var sentence in Sentences)
        cleanText += sentence.Text + " ";

      return cleanText.Trim();
    }

    public void Write()
    {
      Console.Write(GetCleanText());
    }

    public void WriteLine()
    {
      Write();
      Console.WriteLine();
    }

    public void WriteRaw()
    {
      Console.Write(RawText);
    }

    public void WriteLineRaw()
    {
      WriteRaw();
      Console.WriteLine();
    }

    public override string ToString()
    {
      return GetCleanText();
    }
  }

  public class Sentence
  {
    // Member variables (fields, constants):
    protected static int _count = 0;

    private readonly Lazy<List<Phrase>> _prePreTerminalPhrases; // Lazy load pre-preterminal phrases.

    private readonly Lazy<List<Phrase>> _clauses; // Lazy load clauses.

    // Constructors and finalizers:
    protected Sentence()
    {
      Id = _count++;
    }

    public Sentence(string sentence, string lineBreakReplacement = " ", string ignore = "", string punctuation = null, AnalyzerOptions options = AnalyzerOptions.None, Document containingDocument = null)
      : this()
    {
      ContainingDocument = containingDocument;

      if (lineBreakReplacement == null)
        Text = sentence;
      else
        Text = sentence.ReplaceLineBreaks(lineBreakReplacement);

      // Get parse tree:
      // Tree = Repository.StanfordParser.apply(Text).skipRoot(); # 'apply' no longer takes a string parameter [16 Dec. 2014].
      Tree = Repository.StanfordParser.apply(PTBTokenizer.newPTBTokenizer(new java.io.StringReader(Text)).tokenize()).skipRoot();

      // Separate all tokens (OpenNLP version):
      /*
      string[] tokens = Repository.Tokenizer.tokenize(_sentence);
      _tokens = new List<Token>();
      foreach (var token in tokens)
        Tokens.Add(new Token(token));
       */

      // Separate all tokens (Stanford Parser version):
      Tokens = Tree.GetTokens().Tokens;

      // Extract all phrases:
      Phrases = Tree.GetPhrases(this, ignore, punctuation, options);
      _prePreTerminalPhrases = new Lazy<List<Phrase>>(GetPrePreTerminalPhrases);
      //Phrases = Tree.GetClauses(this, ignore, punctuation, options);
      _clauses = new Lazy<List<Phrase>>(GetClauses);

      AddPunctuationDelimitedPhrases(Phrases, ignore, punctuation, options);
    }

    // Properties:
    public Document ContainingDocument
    {
      get; protected set;
    }

    public int Id
    {
      get; protected set;
    }

    public string Text
    {
      get; protected set;
    }

    public List<Token> Tokens
    {
      get; protected set;
    }

    public Tree Tree
    {
      get; protected set;
    }

    public List<Phrase> Phrases
    {
      get; protected set;
    }

    public List<Phrase> PrePreTerminalPhrases
    {
      get { return _prePreTerminalPhrases.Value; }
    }

    public List<Phrase> Clauses
    {
      get { return _clauses.Value; }
    }

    // Methods:
    public void AddPunctuationDelimitedPhrases(List<Phrase> phrases, string ignore = "", string punctuation = null, AnalyzerOptions options = AnalyzerOptions.None)
    {
      // Find phrases based solely on punctutation:
      bool usePunctuationDelimitedPhrases = options.HasFlag(AnalyzerOptions.UsePunctuationDelimitedPhrases);
      if (usePunctuationDelimitedPhrases) {
        phrases.AddRange(GetPunctuationDelimitedPhrases(phrases, ignore, punctuation));

        // Find clauses delimited by just colons or semicolons:
        var clauseIgnore = "#|\\$|\\.|,|\\(|\\)|\"|``|''|`|'|\\!|\\?"; // Added '!', [11 Mar. 2015] '?' [9 Apr. 2015].
        if (ignore != string.Empty)
          clauseIgnore = Regex.Match(ignore, @"^\^?(.*?)\$?$").Result("$1") + "|" + clauseIgnore;
        clauseIgnore = "^(" + clauseIgnore + ")$";
        phrases.AddRange(GetPunctuationDelimitedPhrases(phrases, clauseIgnore, "^\\:$"));
      }
    }

    public List<Phrase> GetPrePreTerminalPhrases()
    {
      var prePreTerminalPhrases = Tree.GetPrePreTerminalPhrases(this, ContainingDocument.Ignore, ContainingDocument.Punctuation, ContainingDocument.Options);

      return prePreTerminalPhrases;
    }

    public List<Phrase> GetClauses()
    {
      var clauses = Tree.GetClauses(this, ContainingDocument.Ignore, ContainingDocument.Punctuation, ContainingDocument.Options);
      AddPunctuationDelimitedPhrases(clauses, ContainingDocument.Ignore, ContainingDocument.Punctuation, ContainingDocument.Options);

      return clauses;
    }

    protected List<Phrase> GetPunctuationDelimitedPhrases(List<Phrase> currentPhrases, string ignore, string punctuation)
    {
      var tentativePhrases = new List<Phrase>();
      var tentativePhrase = new Phrase(sentence: this);
      foreach (var token in Tokens) {
        if (ignore != string.Empty) {
          if (Regex.IsMatch(token.Word, ignore))
            continue;
        }

        bool isPunctuation = Regex.IsMatch(token.TagEquivalent, punctuation ?? Analyzer.PunctuationPatterns);
        if (isPunctuation) {
          // Phrases here should contain two or more tokens:
          if (tentativePhrase.Count > 1)
            tentativePhrases.Add(tentativePhrase);
          tentativePhrase = new Phrase(sentence: this);
        }
        else
          tentativePhrase.Add(token);
      }
      if (tentativePhrase.Count > 1) // In case of no trailing punctuation
        tentativePhrases.Add(tentativePhrase);

      var phrases = new List<Phrase>();
      foreach (var tp in tentativePhrases) {
        bool include = true;
        foreach (var p in currentPhrases) {
          if (tp.EqualInTokens(p)) {
            include = false;  
            break;
          }
        }
        if (include) {
          phrases.Add(tp);
        }
      }

      return phrases;
    }

    public override string ToString()
    {
      return Text;
    }

    // Operators, indexers, and events:
    public Phrase this[int index]
    {
      get { return Phrases[index]; }

      protected set { Phrases[index] = value; }
    }

  }

  [Serializable]
  public class Token : IEquatable<Token>
  {
    // Member variables (fields, constants):
    public static readonly Dictionary<string, string> EquivalenceClasses = new Dictionary<string,string>()
    {
      { "JJR", "JJ" },
      { "JJS", "JJ" },
      { "NNS", "NN" },
      { "NNP", "NN" },
      { "NNPS", "NN" },
      { "NP-TMP", "NN" },
      { "RBR", "RB" },
      { "RBS", "RB" },
      { "WRB", "RB" },
      { "VBD", "VB" },
      { "VBG", "VB" },
      { "VBN", "VB" },
      { "VBP", "VB" },
      { "VBZ", "VB" },
      { "WP$", "WP" },
      { "PRP", "WP" },
      { "PRP$", "WP" },
    };

    private readonly Lazy<List<string>> _derivationalForms; // Lazy load derivationally related word forms from WordNet.

    // Constructors and finalizers:
    protected Token()
    {
      _derivationalForms = new Lazy<List<string>>(FindDerivationalForms);
    }

    public Token(string token, string tag, int depth, CharacterEdges characterEdges = null)
      : this()
    {
      Word = token.Trim();
      Tag = tag.Trim();
      Depth = depth;
      Edges = characterEdges ?? new CharacterEdges();

      string tagEquivalent;
      if (!EquivalenceClasses.TryGetValue(tag, out tagEquivalent)) // cf. 'ContainsKey'
        tagEquivalent = tag;
      TagEquivalent = tagEquivalent;

      Stem = Repository.PorterStemmer.stemTerm(Word);
    }

    // Copy constructor
    public Token(Token token)
      : this()
    {
      Word = token.Word;
      Tag = token.Tag;
      TagEquivalent = token.TagEquivalent;
      Depth = token.Depth;
      Edges = new CharacterEdges(token.Edges);
      Stem = token.Stem;
    }

    // Properties:
    public string Word
    {
      get; protected set;
    }

    public string Tag
    {
      get; protected set;
    }

    public string TagEquivalent
    {
      get; protected set;
    }

    public int Depth
    {
      get; protected set;
    }

    public CharacterEdges Edges
    {
      get; protected set;
    }

    public int Left
    {
      get { return Edges.Left; }
    }

    public int Right
    {
      get { return Edges.Right; }
    }

    public string Stem
    {
      get; protected set;
    }

    public List<string> DerivationalForms
    {
      get { return _derivationalForms.Value; }
    }

    public string Text
    {
      get { return String.Format("{0}/{1}", new object[] { Tag, Word }); }
    }

    public string TextWithDepth
    {
      get { return String.Format("{0}/{1}:{2}", new object[] { Tag, Word, Depth }); }
    }

    public string TextVerbose
    {
      get { return String.Format("{0}/{1}[{3}]:{2}", new object[] { Tag, Word, Depth, Stem }); }
    }

    // Methods:
    public bool IsStopWord(string stopWords = null) // Use "^$" to force no match.
    {
      bool isStopWord = Regex.IsMatch(Word, stopWords ?? Analyzer.StopWords, RegexOptions.IgnoreCase);

      return isStopWord;
    }

    public static List<string> FindDerivationalForms(string startWord, string stem, List<string> prefixes, List<string> suffixes, bool useAllForms = false, bool lowercase = true, bool removeCandidate = true, bool addStem = false, RegexOptions regexOptions = RegexOptions.IgnoreCase)
    { // V. Algorithm 3.3 in Gawryjolek (p. 28)
      var scf = new List<string>();
      var w = startWord.ToLower();
      scf.Add(w);
      var sw = stem.ToLower();
      if (prefixes != null) {
        foreach (var prefix in prefixes) {
          foreach (var word in new List<string>() { w, sw }) {
            var match = Regex.Match(word, "^(" + prefix + ")" + "(?<stem>.*)$", RegexOptions.IgnoreCase);
            if (match.Success) {
              var wnp = match.Groups["stem"].Value;
              var synSets = Repository.WordNet.GetSynSets(wnp);
              if (synSets.Any())
                scf.Add(wnp);
            }
            else {
              var synSets = Repository.WordNet.GetSynSets(prefix + word);
              if (synSets.Any())
                scf.Add(prefix + word);
            }
          }
        }
      }
      if (suffixes != null) {
        foreach (var suffix in suffixes) {
          foreach (var word in new List<string>() { w, sw }) {
            var match = Regex.Match(word, "^(?<stem>.*)" + "(" + suffix + ")$", RegexOptions.IgnoreCase);
            if (match.Success) {
              var wns = match.Groups["stem"].Value;
              var synSets = Repository.WordNet.GetSynSets(wns);
              if (synSets.Any())
                scf.Add(wns);
            }
            else {
              var synSets = Repository.WordNet.GetSynSets(word + suffix);
              if (synSets.Any())
                scf.Add(word + suffix);
            }
          }
        }
      }

      scf = scf.Distinct().ToList(); // Remove duplicates.

      var derivationalForms = new List<string>();
      foreach (var candidate in scf) {
        var synSets = Repository.WordNet.GetSynSets(candidate);
        foreach (var synSet in synSets) {
          var lexicallyRelatedWords = synSet.GetLexicallyRelatedWords();
          if (lexicallyRelatedWords.ContainsKey(WordNetEngine.SynSetRelation.DerivationallyRelated)) {
            var derivationallyRelatedWords = lexicallyRelatedWords[WordNetEngine.SynSetRelation.DerivationallyRelated];
            var keys = new List<string>();
            if (!useAllForms)
              keys.Add(candidate);
            else { // If 'true' use all derivationally related forms.
              foreach (var derivationallyRelatedWord in derivationallyRelatedWords)
                keys.Add(derivationallyRelatedWord.Key);
            }
            foreach (var key in keys) {
              if (derivationallyRelatedWords.ContainsKey(key))
                derivationalForms.AddRange(derivationallyRelatedWords[key]);
            }
          }
        }
      }

      scf.AddRange(derivationalForms);

      for (int i = 0; i < scf.Count; ++i) {
        // Remove underscores from phrases.
        scf[i] = Regex.Replace(scf[i], "_", " ");

        // Remove parenthetical descriptors from words or phrases.
        var match = Regex.Match(scf[i], @"^(.*?)\(.*?\)(.*?)");
        if (match.Success)
          scf[i] = match.Result("$1$2");
      }

      // Convert all synonyms to lowercase.
      if (lowercase)
        scf = scf.ConvertAll(n => n.ToLower());

      scf = scf.Distinct().ToList();

      // Remove candidate word from synonym list.
      if (removeCandidate)
        scf.RemoveAll(n => Regex.IsMatch(n, Regex.Escape(startWord), regexOptions));

      // Add prefixes and suffixes to each derived form and check for the new words' existence in WordNet.
      var additionalForms = new List<string>();
      if (suffixes != null) {
        foreach (var suffix in suffixes) {
          foreach (var candidate in scf) {
            var synSets = Repository.WordNet.GetSynSets(candidate + suffix);
            if (synSets.Any())
              additionalForms.Add(candidate + suffix);
          }
        }
      }
      if (prefixes != null) {
        foreach (var prefix in prefixes) {
          foreach (var candidate in scf) {
            var synSets = Repository.WordNet.GetSynSets(prefix + candidate);
            if (synSets.Any())
              additionalForms.Add(prefix + candidate);
          }
        }
      }

      scf.AddRange(additionalForms);

      if (addStem)
        scf.Add(stem);

      scf = scf.Distinct().ToList(); // Remove duplicates.

      return scf;
    }

    public static List<string> FindDerivationalForms(List<string> words, List<string> prefixes, List<string> suffixes, bool useAllForms = false)
    {
      var derivationalForms = new List<string>();

      foreach (var word in words) {
        var stem = Repository.PorterStemmer.stemTerm(word);
        derivationalForms.AddRange(FindDerivationalForms(word, stem, prefixes, suffixes, useAllForms));
      }

      return derivationalForms;
    }

    public List<string> FindDerivationalForms()
    {
      return FindDerivationalForms(this.Word, this.Stem, Analyzer.MostCommonPrefixes, Analyzer.MostCommonSuffixes, removeCandidate: false, addStem: true);
    }

    public static List<string> FindSynonyms(string word, bool lowercase = true, bool removeCandidate = true, RegexOptions regexOptions = RegexOptions.IgnoreCase)
    {
      var synSets = Repository.WordNet.GetSynSets(word);

      var synonyms = new List<string>();

      foreach (var synSet in synSets) {
        synonyms.AddRange(synSet.Words);
        foreach (var relatedSynSet in synSet.GetRelatedSynSets(WordNetEngine.SynSetRelation.SimilarTo, false)) // Set 'recursive: true' to collect all related SynSets.
          synonyms.AddRange(relatedSynSet.Words);
      }

      for (int i = 0; i < synonyms.Count; ++i) {
        // Remove underscores from phrases.
        synonyms[i] = Regex.Replace(synonyms[i], "_", " ");

        // Remove parenthetical descriptors from words or phrases.
        var match = Regex.Match(synonyms[i], @"^(.*?)\(.*?\)(.*?)");
        if (match.Success)
          synonyms[i] = match.Result("$1$2");
      }

      // Convert all synonyms to lowercase.
      if (lowercase)
        synonyms = synonyms.ConvertAll(n => n.ToLower());

      synonyms = synonyms.Distinct().ToList();

      // Remove candidate word from synonym list.
      if (removeCandidate)
        synonyms.RemoveAll(n => Regex.IsMatch(n, Regex.Escape(word), regexOptions));

      return synonyms;
    }

    public static List<string> FindSynonyms(List<string> words, bool lowercase = true, bool removeCandidate = true, RegexOptions regexOptions = RegexOptions.IgnoreCase)
    {
      var synonyms = new List<string>();

      foreach (var word in words)
        synonyms.AddRange(FindSynonyms(word, lowercase, removeCandidate, regexOptions));

      synonyms = synonyms.Distinct().ToList();

      return synonyms;
    }

    public List<string> FindSynonyms()
    {
      return FindSynonyms(Word);
    }

    public static List<string> FindAntonyms(string word, bool lowercase = true, bool removeCandidate = true, RegexOptions regexOptions = RegexOptions.IgnoreCase)
    {
      var synSets = Repository.WordNet.GetSynSets(word);

      var antonyms = new List<string>();

      foreach (var synSet in synSets) {
        var lexicallyRelatedWords = synSet.GetLexicallyRelatedWords();
        if (lexicallyRelatedWords.ContainsKey(WordNetEngine.SynSetRelation.Antonym)) {
          var antonymWords = lexicallyRelatedWords[WordNetEngine.SynSetRelation.Antonym];
          var keys = new List<string>();
          foreach (var antonymWord in antonymWords)
            keys.Add(antonymWord.Key);
          foreach (var key in keys) {
            if (antonymWords.ContainsKey(key))
              antonyms.AddRange(antonymWords[key]);
          }
        }
        foreach (var relatedSynSet in synSet.GetRelatedSynSets(WordNetEngine.SynSetRelation.Antonym, false)) // Set 'recursive: true' to collect all related SynSets.
          antonyms.AddRange(relatedSynSet.Words);
      }

      for (int i = 0; i < antonyms.Count; ++i) {
        // Remove underscores from phrases.
        antonyms[i] = Regex.Replace(antonyms[i], "_", " ");

        // Remove parenthetical descriptors from words or phrases.
        var match = Regex.Match(antonyms[i], @"^(.*?)\(.*?\)(.*?)");
        if (match.Success)
          antonyms[i] = match.Result("$1$2");
      }

      // Convert all synonyms to lowercase.
      if (lowercase)
        antonyms = antonyms.ConvertAll(n => n.ToLower());

      antonyms = antonyms.Distinct().ToList();

      // Remove candidate word from synonym list.
      if (removeCandidate)
        antonyms.RemoveAll(n => Regex.IsMatch(n, Regex.Escape(word), regexOptions));

      return antonyms;
    }

    public static List<string> FindAntonyms(List<string> words, bool lowercase = true, bool removeCandidate = true, RegexOptions regexOptions = RegexOptions.IgnoreCase)
    {
      var antonyms = new List<string>();

      foreach (var word in words)
        antonyms.AddRange(FindAntonyms(word, lowercase, removeCandidate, regexOptions));

      antonyms = antonyms.Distinct().ToList();

      return antonyms;
    }

    public List<string> FindAntonyms()
    {
      return FindAntonyms(Word);
    }

    public bool EqualTagEquivalent(Token t)
    {
      return (TagEquivalent == t.TagEquivalent);
    }

    public static bool EqualTagEquivalent(Token t1, Token t2)
    {
      return (t1.TagEquivalent == t2.TagEquivalent);
    }

    public bool EqualTagAndDepth(Token t)
    {
      return (TagEquivalent == t.TagEquivalent) && (Depth == t.Depth);
    }

    public static bool EqualTagAndDepth(Token t1, Token t2)
    {
      return (t1.TagEquivalent == t2.TagEquivalent) && (t1.Depth == t2.Depth);
    }

    public bool EqualInWords(Token t, RegexOptions options = RegexOptions.IgnoreCase)
    {
      return Regex.IsMatch(Word, "^" + Regex.Escape(t.Word) + "$", options);
    }

    public static bool EqualInWords(Token t1, Token t2, RegexOptions options = RegexOptions.IgnoreCase)
    {
      return Regex.IsMatch(t1.Word, "^" + Regex.Escape(t2.Word) + "$", options);
    }

    public override bool Equals(Object other) // V. Albahari 2010, pp. 245ff.
    {
      if (!(other is Token)) return false;

      return Equals((Token)other);
    }

    public bool Equals(Token other) // Implements IEquatable<Token>
    {
      return (Word == other.Word) && (Tag == other.Tag) && (Depth == other.Depth) && (Edges == other.Edges);
    }

    public override int GetHashCode() // Must override along with Equals()
    {
      return Word.GetHashCode() ^ Tag.GetHashCode() ^ Depth.GetHashCode() ^ Edges.GetHashCode();
    }

    public override string ToString()
    {
      return Text;
    }

    // Operators, indexers, and events:
    public static bool operator ==(Token t1, Token t2)
    {
      return t1.Equals(t2);
    }

    public static bool operator !=(Token t1, Token t2)
    {
      return !(t1 == t2);
    }
  }

  [Serializable]
  public class CharacterEdges : IEquatable<CharacterEdges>
  {
    // Member variables (fields, constants):

    // Constructors and finalizers:
    public CharacterEdges()
    {
      Left = 0;
      Right = 0;
    }

    public CharacterEdges(int left, int right)
    {
      Left = left;
      Right = right;
    }

    // Copy constructor
    public CharacterEdges(CharacterEdges characterEdges)
    {
      Left = characterEdges.Left;
      Right = characterEdges.Right;
    }

    // Properties:
    public int Left
    {
      get; protected set;
    }

    public int Right
    {
      get; protected set;
    }

    public string Text
    {
      get { return String.Format("Left: {0}; Right: {1}", new object[] { Left, Right }); }
    }

    // Methods:
    public override bool Equals(Object other) // V. Albahari 2010, pp. 245ff.
    {
      if (!(other is CharacterEdges)) return false;

      return Equals((CharacterEdges)other);
    }

    public bool Equals(CharacterEdges other) // Implements IEquatable<CharacterEdges>
    {
      return (Left == other.Left) && (Right == other.Right);
    }

    public override int GetHashCode() // Must override along with Equals()
    {
      return Left.GetHashCode() ^ Right.GetHashCode();
    }

    public override string ToString()
    {
      return Text;
    }

    // Operators, indexers, and events:
    public static bool operator ==(CharacterEdges ce1, CharacterEdges ce2)
    {
      return ce1.Equals(ce2);
    }

    public static bool operator !=(CharacterEdges ce1, CharacterEdges ce2)
    {
      return !(ce1 == ce2);
    }
  }

  [Serializable]
  public class SubsequenceToken : Token
  {
    // Member variables (fields, constants):
    Sentence _sentence = null;

    // Constructors and finalizers:
    public SubsequenceToken(Token token) : base(token)
    {
      IsStart = false; IsEnd = false;
    }

    public SubsequenceToken(Token token, Sentence sentence) : this(token)
    {
      _sentence = sentence;
    }

    // Enums, Structs, and Classes:

    // Properties:
    public bool IsStart
    {
      get; set;
    }

    public bool IsEnd
    {
      get; set;
    }

    public Sentence ContainingSentence
    {
      get { return _sentence; }
    }

    public int SentenceId
    {
      get { return ContainingSentence.Id; }
    }

    // Methods:

    // Operators, indexers, and events:
    public override bool Equals(Object other) // V. Albahari 2010, pp. 245ff.
    {
      if (!(other is SubsequenceToken)) return false;

      return Equals((SubsequenceToken)other);
    }

    public bool Equals(SubsequenceToken other) // Implements IEquatable<SubsequenceToken>
    {
      return (Word == other.Word) && (Tag == other.Tag) && (Depth == other.Depth) && (Edges == other.Edges) && (SentenceId == other.SentenceId);
    }

    public override int GetHashCode() // Must override along with Equals()
    {
      return Word.GetHashCode() ^ Tag.GetHashCode() ^ Depth.GetHashCode() ^ Edges.GetHashCode() ^ SentenceId.GetHashCode();
    }

  }

  class SubsequenceTokenComparer : IEqualityComparer<SubsequenceToken>
  {
    public bool Equals(SubsequenceToken t1, SubsequenceToken t2)
    {
      if (Object.ReferenceEquals(t1, t2))
        return true; // Identical is a fortiori equal.

      if (Object.ReferenceEquals(t1, null) || Object.ReferenceEquals(t2, null))
        return false;

      if ((t1.Edges == t2.Edges) && t1.SentenceId == t2.SentenceId)
        return true;

      return false;
    }

    public int GetHashCode(SubsequenceToken token)
    {
      if (Object.ReferenceEquals(token, null))
        return 0;

      return token.Word.GetHashCode();
    }
  }

  class SubsequenceTokenEquivalenceComparer : IEqualityComparer<SubsequenceToken>
  {
    public bool Equals(SubsequenceToken t1, SubsequenceToken t2)
    {
      if (Object.ReferenceEquals(t1, t2))
        return false; // We want matched tokens that aren't identical!

      if (Object.ReferenceEquals(t1, null) || Object.ReferenceEquals(t2, null))
        return false;

      if ((t1.Edges == t2.Edges) && t1.SentenceId == t2.SentenceId)
        return false;

      return SubsequenceToken.EqualInWords(t1, t2);
    }

    public int GetHashCode(SubsequenceToken token)
    {
      if (Object.ReferenceEquals(token, null))
        return 0;

      return token.Word.GetHashCode();
    }
  }

  class SubsequenceTokenTagEquivalentEquivalenceComparer : IEqualityComparer<SubsequenceToken>
  {
    public bool Equals(SubsequenceToken t1, SubsequenceToken t2)
    {
      if (Object.ReferenceEquals(t1, t2))
        return false; // We want matched tokens that aren't identical!

      if (Object.ReferenceEquals(t1, null) || Object.ReferenceEquals(t2, null))
        return false;

      if ((t1.Edges == t2.Edges) && t1.SentenceId == t2.SentenceId)
        return false;

      return SubsequenceToken.EqualTagEquivalent(t1, t2);
    }

    public int GetHashCode(SubsequenceToken token)
    {
      if (Object.ReferenceEquals(token, null))
        return 0;

      return token.Word.GetHashCode();
    }
  }

  [Serializable]
  public class Subsequence : List<SubsequenceToken>
  {
    // Member variables (fields, constants):
    Sentence _sentence = null;
    Subsequence _containingSubsequence = null;
    int _windowId = -1;
    StopWordsOptions _stopWordsStatus = StopWordsOptions.None;

    // Constructors and finalizers:
    public Subsequence()
    {
    }

    public Subsequence(IEnumerable<SubsequenceToken> collection)
      : base(collection)
    {
    }

    public Subsequence(IEnumerable<SubsequenceToken> collection, int windowId)
      : this(collection)
    {
      _windowId = windowId;
    }

    public Subsequence(IEnumerable<SubsequenceToken> collection, Sentence sentence, Subsequence containingSubsequence)
      : this(collection)
    {
      _sentence = sentence;
      _containingSubsequence = containingSubsequence;
    }

    public Subsequence(IEnumerable<SubsequenceToken> collection, Sentence sentence, Subsequence containingSubsequence, int windowId)
      : this(collection, sentence, containingSubsequence)
    {
      _windowId = windowId;
    }

    public Subsequence(Subsequence collection)
      : this(collection, collection.ContainingSentence, collection.ContainingSubsequence, collection.WindowId)
    {
    }

    public Subsequence(Subsequence collection, int windowId)
      : this(collection, collection.ContainingSentence, collection.ContainingSubsequence, windowId)
    {
    }

    // Enums, Structs, and Classes:

    // Properties:
    public Sentence ContainingSentence
    {
      get { return _sentence; }
    }

    public int SentenceId
    {
      get { return ContainingSentence.Id; }
    }

    public Subsequence ContainingSubsequence
    {
      get { return _containingSubsequence; }
      set { _containingSubsequence = value; }
    }

    public int WindowId
    {
      get { return _windowId; }
    }

    public StopWordsOptions StopWordsStatus
    {
      get { return _stopWordsStatus; }
      set { _stopWordsStatus = value; }
    }

    // Methods:
    public int? NextLeftEdge() // Return null if subsequence is already at left edge of containing subsequence.
    {
      if (ContainingSubsequence != null) {
        var leftElement = this.First();

        var containedMatchIndex = ContainingSubsequence.FindIndex(
          delegate(SubsequenceToken st)
          {
            return st.Edges == leftElement.Edges;
          }
        );

        if (containedMatchIndex != -1 && containedMatchIndex != 0)
          return ContainingSubsequence[containedMatchIndex - 1].Right;
      }

      return null;
    }

    // Q: Should these functions test for reference equality in the delegate instead? I think that's less robust than checking the edges; the edges define a token uniquely in any subsequence, so identification by edge equality should never fail.

    public SubsequenceToken NextLeft() // Return null if subsequence is already at left edge of containing subsequence.
    {
      if (!Object.ReferenceEquals(null, ContainingSubsequence)) { // 'Object.ReferenceEquals' because 'Equals' throws an error.
        var leftElement = this.First();

        var containedMatchIndex = ContainingSubsequence.FindIndex(
          delegate(SubsequenceToken st)
          {
            return st.Edges == leftElement.Edges;
          }
        );

        if (containedMatchIndex != -1 && containedMatchIndex != 0)
          return ContainingSubsequence[containedMatchIndex - 1];
      }

      return null;
    }

    public int? NextRightEdge() // Return null if subsequence is already at right edge of containing subsequence.
    {
      if (ContainingSubsequence != null)
      {
        var rightElement = this.Last();

        var containedMatchIndex = ContainingSubsequence.FindIndex(
          delegate(SubsequenceToken st)
          {
            return st.Edges == rightElement.Edges;
          }
        );

        if (containedMatchIndex != -1 && containedMatchIndex != ContainingSubsequence.Count - 1)
          return ContainingSubsequence[containedMatchIndex + 1].Left;
      }

      return null;
    }

    public SubsequenceToken NextRight() // Return null if subsequence is already at right edge of containing subsequence.
    {
      if (!Object.ReferenceEquals(null, ContainingSubsequence)) { // 'Object.ReferenceEquals' because 'Equals' throws an error.
        var rightElement = this.Last();

        var containedMatchIndex = ContainingSubsequence.FindIndex(
          delegate(SubsequenceToken st)
          {
            return st.Edges == rightElement.Edges;
          }
        );

        if (containedMatchIndex != -1 && containedMatchIndex != ContainingSubsequence.Count - 1)
          return ContainingSubsequence[containedMatchIndex + 1];
      }

      return null;
    }

    public bool IsLeftContiguous(Subsequence s) // I.e., does 's' end with the token immediately to the left of 'this' within the containing subsequence?
    {
      bool isLeftContiguous = false;

      if (SentenceId == s.SentenceId) {
        var nextLeft = this.NextLeft();
        if (!Object.ReferenceEquals(null, nextLeft)) {
          if (s.Last() == nextLeft)
            isLeftContiguous = true;
        }
      }
      else if (s.SentenceId + 1 == SentenceId) {
        if (Object.ReferenceEquals(null, s.NextRight()) && Object.ReferenceEquals(null, this.NextLeft()))
          isLeftContiguous = true;
      }

      return isLeftContiguous;
    }

    public bool IsRightContiguous(Subsequence s) // I.e., does 's' start with the token immediately to the right of 'this' within the containing subsequence?
    {
      bool isRightContiguous = false;

      if (SentenceId == s.SentenceId) {
        var nextRight = this.NextRight();
        if (!Object.ReferenceEquals(null, nextRight)) {
          if (s.First() == nextRight)
            isRightContiguous = true;
        }
      }
      else if (SentenceId + 1 == s.SentenceId) {
        if (Object.ReferenceEquals(null, s.NextLeft()) && Object.ReferenceEquals(null, this.NextRight()))
            isRightContiguous = true;
      }

      return isRightContiguous;
    }

    public bool IsContiguous(Subsequence s)
    {
      return IsLeftContiguous(s) || IsRightContiguous(s);
    }

    public delegate Subsequence BacktrackDelegateSimple(ref int[,] c, ref Subsequence p1, ref Subsequence p2, int i, int j);
    public static int LcsLengthSimple(Subsequence p1, Subsequence p2, bool backtrack = false)
    { // See: http://en.wikipedia.org/wiki/Longest_common_subsequence_problem
      int m = p1.Count, n = p2.Count;
      int[,] c = new int[m + 1, n + 1]; // 'i's are initialized to 0
      for (int i = 1; i < c.GetLength(0); ++i) {
        for (int j = 1; j < c.GetLength(1); ++j) {
          if (p1[i - 1].TagEquivalent == p2[j - 1].TagEquivalent)
            c[i, j] = c[i - 1, j - 1] + 1;
          else
            c[i, j] = Math.Max(c[i, j - 1], c[i - 1, j]);
        }
      }
      int lcsLength = c[m, n];

      BacktrackDelegateSimple Backtrack = delegate(ref int[,] C, ref Subsequence P1, ref Subsequence P2, int i, int j)
      {
        MethodBase method = new StackTrace().GetFrame(0).GetMethod();

        if (i == 0 || j == 0)
          return new Subsequence();
        else if (P1[i - 1].TagEquivalent == P2[j - 1].TagEquivalent) {
          var temp = (Subsequence)method.Invoke(null, new object[] { C, P1, P2, i - 1, j - 1 });
          temp.Add(P1[i - 1]);

          return temp;
        } else {
          if (C[i, j - 1] > C[i - 1, j])
            return (Subsequence)method.Invoke(null, new object[] { C, P1, P2, i, j - 1 });
          else
            return (Subsequence)method.Invoke(null, new object[] { C, P1, P2, i - 1, j });
        }
      };

      Subsequence backtrackList;
      if (backtrack) {
        backtrackList = Backtrack(ref c, ref p1, ref p2, m, n);

        Console.WriteLine("Longest common subsequence:");
        Console.WriteLine(backtrackList);
      }

      return lcsLength;
    }

    public delegate List<Subsequence> BacktrackDelegate(ref int[,] c, ref Subsequence p1, ref Subsequence p2, int i, int j);
    public static int LcsLength(Subsequence p1, Subsequence p2, bool backtrack = false)
    { // See: http://en.wikipedia.org/wiki/Longest_common_subsequence_problem
      int m = p1.Count, n = p2.Count;
      int[, ] c = new int[m + 1, n + 1]; // 'i's are initialized to 0
      for (int i = 1; i < c.GetLength(0); ++i) {
        for (int j = 1; j < c.GetLength(1); ++j) {
          if (p1[i - 1].TagEquivalent == p2[j - 1].TagEquivalent)
            c[i, j] = c[i - 1, j - 1] + 1;
          else
            c[i, j] = Math.Max(c[i, j - 1], c[i - 1, j]);
        }
      }
      int lcsLength = c[m, n];

      BacktrackDelegate Backtrack = delegate(ref int[,] C, ref Subsequence P1, ref Subsequence P2, int i, int j)
      {
        MethodBase method = new StackTrace().GetFrame(0).GetMethod();

        if (i == 0 || j == 0)
          return new List<Subsequence>();
        else if (P1[i - 1].TagEquivalent == P2[j - 1].TagEquivalent) {
          var temp = (List<Subsequence>)method.Invoke(null, new object[] { C, P1, P2, i - 1, j - 1 });
          if (temp.Count == 0)
            temp.Add(new Subsequence());

          for (int k = 0; k < temp.Count; ++k)
            temp[k].Add(P1[i - 1]);

          return temp;
        } else {
          var r = new List<Subsequence>();

          if (C[i, j - 1] >= C[i - 1, j]) {
            var temp = (List<Subsequence>)method.Invoke(null, new object[] { C, P1, P2, i, j - 1 });
            r.AddRange(temp);
          }
          if (C[i - 1, j] >= C[i, j - 1]) {
            var temp = (List<Subsequence>)method.Invoke(null, new object[] { C, P1, P2, i - 1, j });
            r.AddRange(temp);
          }

          return r;
        }
      };

      List<Subsequence> backtrackList = null;
      if (backtrack) {
        IEnumerable<Subsequence> distinct = Backtrack(ref c, ref p1, ref p2, m, n).Distinct(/*new SubsequenceComparer()*/);
        backtrackList = new List<Subsequence>(distinct);

        Console.WriteLine("Longest common subsequences:");
        foreach (var phrase in backtrackList)
          Console.WriteLine(phrase);
      }

      return lcsLength;
    }

    public static int LevenshteinDistance(Subsequence source, Subsequence target)
    { // See: http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance
      if (source.Count > target.Count) {
        var temp = target;
        target = source;
        source = temp;
      }

      int m = target.Count, n = source.Count;
      var distance = new int[2, m + 1];

      // Initialize the distance matrix.
      for (int j = 1; j <= m; j++)
        distance[0, j] = j;

      var currentRow = 0;
      for (int i = 1; i <= n; ++i) {
        currentRow = i & 1;
        distance[currentRow, 0] = i;
        var previousRow = currentRow ^ 1;
        for (int j = 1; j <= m; j++) {
          var cost = (target[j - 1].TagEquivalent == source[i - 1].TagEquivalent ? 0 : 1);
          distance[currentRow, j] = Math.Min(
            Math.Min(distance[previousRow, j] + 1, distance[currentRow, j - 1] + 1),
            distance[previousRow, j - 1] + cost
          );
        }
      }

      return distance[currentRow, m];
    }

    public static int MaximumWordTagOverlap(Subsequence p1, Subsequence p2, bool levenshtein = false) // V. Gawryjolek, pp. 32-36
    {
      if (levenshtein)
        return Math.Max(p1.Count, p2.Count) - LevenshteinDistance(p1, p2);
      else
        return LcsLength(p1, p2);
    }

    public bool Equivalent(Subsequence other) // Implements IEquatable<Subsequence>
    {
      if (Object.ReferenceEquals(this, other))
        return false; // We want matched subsequences that aren't identical!

      if (Object.ReferenceEquals(this, null) || Object.ReferenceEquals(other, null))
        return false;

      if (this.Count != other.Count)
        return false;

      return this.SequenceEqual(other, new SubsequenceTokenEquivalenceComparer());
    }

    public bool EqualsInTagEquivalent(Subsequence other) // Implements IEquatable<Subsequence>
    {
      if (Object.ReferenceEquals(this, other))
        return false; // We want matched subsequences that aren't identical!

      if (Object.ReferenceEquals(this, null) || Object.ReferenceEquals(other, null))
        return false;

      if (this.Count != other.Count)
        return false;

      return this.SequenceEqual(other, new SubsequenceTokenTagEquivalentEquivalenceComparer());
    }

    public override bool Equals(Object other) // V. Albahari 2010, pp. 245ff.
    {
      if (!(other is Subsequence)) return false;

      return Equals((Subsequence)other);
    }

    public bool Equals(Subsequence other)
    {
      if (Object.ReferenceEquals(this, other))
        return true; // Identical is a fortiori equal.

      if (Object.ReferenceEquals(this, null) || Object.ReferenceEquals(other, null))
        return false;

      if (this.Count != other.Count)
        return false;

      return this.SequenceEqual(other, new SubsequenceTokenComparer());
    }

    public override int GetHashCode()
    {
      int tokensHashCode = 0;
      foreach (var token in this)
        tokensHashCode ^= token.GetHashCode();

      return tokensHashCode;
    }

    // Operators, indexers, and events:
    public static int operator -(Subsequence p1, Subsequence p2) // V. Gawryjolek, pp. 32-36
    {
      int l1 = p1.Count, l2 = p2.Count;
      int d = Math.Abs(l1 - l2);

      for (int i = 0; i < Math.Min(l1, l2); ++i) {
        if (Token.EqualTagAndDepth(p1[i], p2[i]))
          continue;
        else {
          d += Math.Min(l1, l2) - MaximumWordTagOverlap(p1, p2, false); // If 'true', use Levenshtein distance instead of LCS length.

          return d;
        }
      }

      return d;
    }

    public static bool operator ==(Subsequence s1, Subsequence s2)
    {
      return s1.Equals(s2);
    }

    public static bool operator !=(Subsequence s1, Subsequence s2)
    {
      return !(s1 == s2);
    }
  }

  class SubsequenceComparer : IEqualityComparer<Subsequence>
  {
    public bool Equals(Subsequence s1, Subsequence s2)
    {
      if (Object.ReferenceEquals(s1, s2))
        return true;

      if (Object.ReferenceEquals(s1, null) || Object.ReferenceEquals(s2, null))
        return false;

      if (s1.Count != s2.Count)
        return false;

      return s1.SequenceEqual(s2, new SubsequenceTokenComparer());
    }

    public int GetHashCode(Subsequence subsequence)
    {
      int tokensHashCode = 0;
      foreach (var token in subsequence)
        tokensHashCode ^= token.GetHashCode();

      return tokensHashCode;
    }
  }

  [Serializable]
  public class Phrase : IEquatable<Phrase>
  {
    // Member variables (fields, constants):
    private Sentence _sentence = null; // Reference to containing 'Sentence' object if applicable.

    private readonly Lazy<List<Subsequence>> _subsequences; // Lazy load the contiguous subsequences.
    private readonly Lazy<List<Subsequence>> _subsequencesNoStopWords;
    private readonly Lazy<List<Subsequence>> _subsequencesNoBoundaryDeterminersEtc;
    private readonly Lazy<List<Subsequence>> _subsequencesNoDeterminersEtc;
    private readonly Lazy<List<Subsequence>> _subsequencesNoStartDeterminersEtc;
    private readonly Lazy<List<Subsequence>> _subsequencesKeepNounsVerbsAdjectivesAdverbsPronounsTagEquivalent;
    private readonly Lazy<List<Subsequence>> _subsequencesKeepNounsVerbsAdjectivesAdverbsTag;
    private readonly Lazy<List<Subsequence>> _subsequencesNoBoundaryConjunctions;

    // Constructors and finalizers:
    public Phrase()
    {
      _subsequences = new Lazy<List<Subsequence>>(GetSubsequences);
      _subsequencesNoStopWords = new Lazy<List<Subsequence>>(GetSubsequencesNoStopWords);
      _subsequencesNoDeterminersEtc = new Lazy<List<Subsequence>>(GetSubsequencesNoDeterminersEtc);
      _subsequencesNoBoundaryDeterminersEtc = new Lazy<List<Subsequence>>(GetSubsequencesNoBoundaryDeterminersEtc);
      _subsequencesNoStartDeterminersEtc = new Lazy<List<Subsequence>>(GetSubsequencesNoStartDeterminersEtc);
      _subsequencesKeepNounsVerbsAdjectivesAdverbsPronounsTagEquivalent = new Lazy<List<Subsequence>>(GetSubsequencesKeepNounsVerbsAdjectivesAdverbsPronounsTagEquivalent);
      _subsequencesKeepNounsVerbsAdjectivesAdverbsTag = new Lazy<List<Subsequence>>(GetSubsequencesKeepNounsVerbsAdjectivesAdverbsTag);
      _subsequencesNoBoundaryConjunctions = new Lazy<List<Subsequence>>(GetSubsequencesNoBoundaryConjunctions);
    }

    public Phrase(bool isPunctuationOmitted = false, Sentence sentence = null)
      : this()
    {
      Tokens = new List<Token>();
      IsPunctuationOmitted = isPunctuationOmitted;
      _sentence = sentence;
    }

    public Phrase(List<Token> tokens, bool isPunctuationOmitted = false, Sentence sentence = null)
      : this()
    {
      Tokens = tokens;
      IsPunctuationOmitted = isPunctuationOmitted;
      _sentence = sentence;
    }

    public Phrase(Phrase phrase)
      : this()
    {
      Tokens = phrase.Tokens;
      IsPunctuationOmitted = phrase.IsPunctuationOmitted;
      _sentence = phrase.ContainingSentence;
    }

    // Properties:
    public List<Token> Tokens
    {
      get; protected set;
    }

    public bool IsPunctuationOmitted
    {
      get; set;
    }

    public int Left
    {
      get { return Tokens.First().Left; }
    }

    public int Right
    {
      get { return Tokens.Last().Right; }
    }

    public Sentence ContainingSentence
    {
      get { return _sentence; }
    }

    public int SentenceId
    {
      get { return ContainingSentence.Id; }
    }

    public List<Subsequence> Subsequences
    {
      get { return _subsequences.Value; }
    }

    public List<Subsequence> SubsequencesNoStopWords
    {
      get { return _subsequencesNoStopWords.Value; }
    }

    public List<Subsequence> SubsequencesNoDeterminersEtc
    {
      get { return _subsequencesNoDeterminersEtc.Value; }
    }

    public List<Subsequence> SubsequencesNoBoundaryDeterminersEtc
    {
      get { return _subsequencesNoBoundaryDeterminersEtc.Value; }
    }

    public List<Subsequence> SubsequencesNoStartDeterminersEtc
    {
      get { return _subsequencesNoStartDeterminersEtc.Value; }
    }

    public List<Subsequence> SubsequencesKeepNounsVerbsAdjectivesAdverbsPronounsTagEquivalent
    {
      get { return _subsequencesKeepNounsVerbsAdjectivesAdverbsPronounsTagEquivalent.Value; }
    }

    public List<Subsequence> SubsequencesKeepNounsVerbsAdjectivesAdverbsTag
    {
      get { return _subsequencesKeepNounsVerbsAdjectivesAdverbsTag.Value; }
    }

    public List<Subsequence> SubsequencesNoBoundaryConjunctions
    {
      get { return _subsequencesNoBoundaryConjunctions.Value; }
    }

    public string Text
    {
      get
      {
        string s = string.Empty;
        foreach (var token in Tokens)
          s += token.Text + " ";

        return s.Trim();
      }
    }

    public string TextWithDepth
    {
      get
      {
        string s = string.Empty;
        foreach (var token in Tokens)
          s += token.TextWithDepth + " ";

        return s.Trim();
      }
    }

    public string TextVerbose
    {
      get
      {
        string s = string.Empty;
        foreach (var token in Tokens)
          s += token.TextVerbose + " ";

        return s.Trim();
      }
    }

    public int Count
    {
      get { return Tokens.Count; }
    }

    // Methods:
    public List<Subsequence> ContiguousSubsequences(string punctuation = null, string stopWords = null, StopWordsOptions options = StopWordsOptions.None, bool tagEquivalent = true) // For either of the first two parameters, use "^$" to force no match.
    {
      var subsequences = new List<Subsequence>();

      // If either of these is true then stop words aren't generally omitted, only at the first and/or last word.
      bool firstWord = options.HasFlag(StopWordsOptions.FirstWord);
      bool lastWord = options.HasFlag(StopWordsOptions.LastWord);
      bool matchTag = options.HasFlag(StopWordsOptions.MatchTag);
      bool keepWords = options.HasFlag(StopWordsOptions.KeepWords);

      var subset = new List<SubsequenceToken>();
      var tokens = new List<Token>(Tokens); // Make shallow copy of 'Tokens'.

      if (keepWords) {
        for (int i = 0; i < tokens.Count; ++i) {
          bool isKeepWord = Regex.IsMatch(matchTag ? (tagEquivalent ? tokens[i].TagEquivalent : tokens[i].Tag) : tokens[i].Word, stopWords ?? @"^$", RegexOptions.IgnoreCase);
          if (!isKeepWord) {
            tokens.RemoveAt(i);
            i -= 1;
            continue;
          }
        }

        stopWords = @"^$";
      }

      var stopWordsStatus = StopWordsOptions.None;
      for (int i = 0; i < tokens.Count; ++i) {
        bool isPunctuation = Regex.IsMatch((tagEquivalent ? tokens[i].TagEquivalent : tokens[i].Tag), punctuation ?? Analyzer.PunctuationPatterns);
        bool isStopWord = Regex.IsMatch(matchTag ? (tagEquivalent ? tokens[i].TagEquivalent : tokens[i].Tag) : tokens[i].Word, stopWords ?? Analyzer.StopWords, RegexOptions.IgnoreCase);
        if (!isPunctuation && !isStopWord)
          subset.Add(new SubsequenceToken(tokens[i], ContainingSentence));
        else if (isStopWord) {
          if (firstWord || lastWord) {
            if (firstWord) {
              if (i == 0) {
                tokens.RemoveAt(i);
                stopWordsStatus |= StopWordsOptions.FirstWord;
                i -= 1;
                continue;
              }
            }
            if (lastWord) {
              if (i == tokens.Count - 1) {
                stopWordsStatus |= StopWordsOptions.LastWord;
                continue;
              }
            }
            subset.Add(new SubsequenceToken(tokens[i], ContainingSentence));
          }
        }
      }
      // Pick up stop-word stragglers at the end of the phrase.
      if (lastWord && subset.Count > 0) {
        int i = subset.Count - 1;
        while (true) {
          bool isStopWord = Regex.IsMatch(matchTag ? (tagEquivalent ? subset[i].TagEquivalent : subset[i].Tag) : subset[i].Word, stopWords ?? Analyzer.StopWords, RegexOptions.IgnoreCase);
          if (isStopWord) {
            subset.RemoveAt(i);
            stopWordsStatus |= StopWordsOptions.LastWord;
            i = subset.Count - 1;
          }
          else
            break;
        }
      }

      if (subset.Count > 0) {
        // Mark start and end tokens of phrase
        subset.First().IsStart = true; subset.Last().IsEnd = true;

        for (int i = 0; i < subset.Count; ++i) {
          for (int j = i; j < subset.Count; ++j) {
            var current = subset.Skip(i).Take(subset.Count - j).ToArray();
            if (i == 0 && j == 0)
              subsequences.Add(new Subsequence(current, ContainingSentence, null));
            else
              subsequences.Add(new Subsequence(current, ContainingSentence, subsequences[0]));

            if (i == 0 && stopWordsStatus.HasFlag(StopWordsOptions.FirstWord))
              subsequences.Last().StopWordsStatus |= StopWordsOptions.FirstWord;

            if (j == 0 && stopWordsStatus.HasFlag(StopWordsOptions.LastWord))
              subsequences.Last().StopWordsStatus |= StopWordsOptions.LastWord;
          }
        }
      }

      return subsequences;
    }

    public List<Subsequence> GetSubsequences()
    {
      return ContiguousSubsequences(stopWords: @"^$");
    }

    public List<Subsequence> GetSubsequencesNoStopWords()
    {
      return ContiguousSubsequences();
    }

    public List<Subsequence> GetSubsequencesNoDeterminersEtc()
    {
      return ContiguousSubsequences(stopWords: Analyzer.DeterminersConjunctionsPrepositionsTag, options: StopWordsOptions.MatchTag);
    }

    public List<Subsequence> GetSubsequencesNoBoundaryDeterminersEtc()
    {
      //return ContiguousSubsequences(stopWords: Analyzer.DeterminersConjunctionsPrepositions, options: StopWordsOptions.FirstWord | StopWordsOptions.LastWord);
      return ContiguousSubsequences(stopWords: Analyzer.DeterminersConjunctionsPrepositionsTag, options: StopWordsOptions.FirstWord | StopWordsOptions.LastWord | StopWordsOptions.MatchTag);
    }

    public List<Subsequence> GetSubsequencesNoStartDeterminersEtc()
    {
      return ContiguousSubsequences(stopWords: Analyzer.DeterminersConjunctionsPrepositionsTag, options: StopWordsOptions.FirstWord | StopWordsOptions.MatchTag);
    }

    public List<Subsequence> GetSubsequencesKeepNounsVerbsAdjectivesAdverbsPronounsTagEquivalent()
    {
      return ContiguousSubsequences(stopWords: Analyzer.NounsVerbsAdjectivesAdverbsPronounsTagEquivalent, options: StopWordsOptions.KeepWords | StopWordsOptions.MatchTag);
    }

    public List<Subsequence> GetSubsequencesKeepNounsVerbsAdjectivesAdverbsTag()
    {
      return ContiguousSubsequences(stopWords: Analyzer.NounsVerbsAdjectivesAdverbsTag, options: StopWordsOptions.KeepWords | StopWordsOptions.MatchTag, tagEquivalent: false);
    }

    public List<Subsequence> GetSubsequencesNoBoundaryConjunctions()
    {
      return ContiguousSubsequences(stopWords: Analyzer.ConjunctionsTag, options: StopWordsOptions.FirstWord | StopWordsOptions.LastWord | StopWordsOptions.MatchTag);
    }

    public void Add(Token token)
    {
      Tokens.Add(token);
    }

    public void Clear()
    {
      Tokens.Clear();
      IsPunctuationOmitted = false;
    }

    public bool EqualExceptPunctuationOmission(Phrase p)
    {
      return (Tokens.SequenceEqual(p.Tokens)) && (IsPunctuationOmitted != p.IsPunctuationOmitted);
    }

    public bool EqualInTokens(Phrase p)
    {
      return Tokens.SequenceEqual(p.Tokens);      
    }

    public override bool Equals(Object other) // V. Albahari 2010, pp. 245ff.
    {
      if (!(other is Phrase)) return false;

      return Equals((Phrase)other);
    }

    public bool Equals(Phrase other) // Implements IEquatable<Phrase>
    {
      return (Tokens.SequenceEqual(other.Tokens)) && (IsPunctuationOmitted == other.IsPunctuationOmitted);
    }

    public override int GetHashCode() // Must override along with Equals()
    {
      int tokensHashCode = 0;
      foreach (var token in Tokens)
        tokensHashCode ^= token.GetHashCode();

      return tokensHashCode ^ IsPunctuationOmitted.GetHashCode();
    }

    public override string ToString()
    {
      return TextWithDepth;
    }

    // Operators, indexers, and events:
    public static bool operator ==(Phrase p1, Phrase p2)
    {
      return p1.Equals(p2);
    }

    public static bool operator !=(Phrase p1, Phrase p2)
    {
      return !(p1 == p2);
    }

    public Token this[int index]
    {
      get { return Tokens[index]; }

      set { Tokens[index] = value; }
    }
  }

  public class PhraseComparer : IEqualityComparer<Phrase>
  {
    bool IEqualityComparer<Phrase>.Equals(Phrase x, Phrase y)
    {
      if (Object.ReferenceEquals(x, y))
        return true;

      if (Object.ReferenceEquals(x, null) || Object.ReferenceEquals(y, null))
        return false;

      return x == y;
    }

    int IEqualityComparer<Phrase>.GetHashCode(Phrase obj)
    {
      if (Object.ReferenceEquals(obj, null))
        return 0;

      return obj.GetHashCode();
    }
  }
}
