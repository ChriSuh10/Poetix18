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
  [Flags]
  public enum RhetoricalFigures
  {
    None          = 0x000000,
    Anadiplosis   = 0x000001, // Repetition of the ending word or phrase from the previous clause at the beginning of the next.
    Anaphora      = 0x000002, // Repetition of a word or phrase at the beginning of successive phrases or clauses.
    Antimetabole  = 0x000004, // Repetition of words in reverse grammatical order.
    Chiasmus      = 0x000008, // Repetition of grammatical structures in reverse order.
    Conduplicatio = 0x000010, // The repetition of a word or phrase.
    Epanalepsis   = 0x000020, // Repetition at the end of a clause of the word or phrase that began it.
    Epistrophe    = 0x000040, // Repetition of the same word or phrase at the end of successive clauses.
    Epizeuxis     = 0x000080, // Repetition of a word or phrase with no others between.
    Isocolon      = 0x000100, // Repetition of grammatical structure in nearby phrases or clauses of approximately equal length.
    Oxymoron      = 0x000200, // A terse paradox; the yoking of two contradictory terms.
    Ploce         = 0x000400, // The repetition of word in a short span of text for rhetorical emphasis.
    Polyptoton    = 0x000800, // Repetition of a word in a different form; having cognate words in close proximity.
    Polysyndeton  = 0x001000, // "Excessive" repetition of conjunctions between clauses.
    Symploce      = 0x002000, // Repetition of a word or phrase at the beginning, and of another at the end, of successive clauses.
    All           = 0x100000
  }

  public class RhetoricalFigure
  {
    // Member variables (fields, constants):
    public const string FigureComponentsSeparator = "|";

    // Constructors and finalizers:
    public RhetoricalFigure()
    {
    }

    public RhetoricalFigure(Subsequence subsequence, RhetoricalFigures type, int windowId)
    {
      Tokens = subsequence;
      Type = type;
      WindowId = windowId;
    }

    // Enums, Structs, and Classes:

    // Properties:
    public RhetoricalFigures Type
    { get; protected set; }

    public int WindowId
    { get; protected set; }

    public List<SubsequenceToken> Tokens
    { get; protected set; }

    // Methods:
    public static List<RhetoricalFigure> MergeFigures(List<List<Subsequence>> subsequences, RhetoricalFigures type, bool multiWindow = true, string demarcation = FigureComponentsSeparator)
    {
      // Some figures may be out of order WRT the start of the text; reorder them here.
      subsequences = subsequences.OrderBy(s => s[0].SentenceId).ThenBy(s => s[0][0].Left).ToList();

      var deepSubsequences = subsequences;

      // Merge multi-window figures.
      if (multiWindow) {
        for (int i = 0; i < deepSubsequences.Count - 1; ++i) {
          for (int j = i + 1; j < deepSubsequences.Count; ++j) {
            var intersection = deepSubsequences[i].Intersect(deepSubsequences[j]);
            if (intersection.Any()) {
              var intersectionList = intersection.ToList();
              if (deepSubsequences[i].Last() == intersectionList.Last() && deepSubsequences[j].First() == intersectionList.First()) {
                var merger = deepSubsequences[i].Union(deepSubsequences[j]).ToList();
                deepSubsequences[i] = merger;
                deepSubsequences.RemoveAt(j);
                i -= 1;
                break;
              }
            }
          }
        }
      }

      // At this point, no subsequence component of any figure should contain part of any other subsequence component.
      for (int i = 0; i < deepSubsequences.Count; ++i)
        deepSubsequences[i] = deepSubsequences[i].Distinct().ToList();

      // N.B. V. the following for a discussion of how 'Distinct()' results are ordered:
      // http://stackoverflow.com/questions/4734852/does-c-sharp-distinct-method-keep-original-ordering-of-sequence-intact

      // Flatten out subsequence lists.
      var flatSubsequences = new List<Subsequence>();
      foreach (var s in deepSubsequences)
        flatSubsequences.Add(new Subsequence(s.SelectMany(x => x), s[0].WindowId));

      // At this point, no subsequence component of any figure should contain part of any other subsequence component.
      //for (int i = 0; i < flatSubsequences.Count; ++i) {
      //  if (flatSubsequences[i].Distinct().Count() < flatSubsequences[i].Count) {
      //    flatSubsequences.RemoveAt(i);
      //    deepSubsequences.RemoveAt(i);
      //    i -= 1;
      //    continue;
      //  }
      //}

      // Remove duplicate list instances and merge those contained in others.
      for (int i = 0; i < flatSubsequences.Count - 1; ++i) {
        for (int j = i + 1; j < flatSubsequences.Count; ++j) {
          if (flatSubsequences[i].IsSupersetOf(flatSubsequences[j])) {
            flatSubsequences.RemoveAt(j);
            deepSubsequences.RemoveAt(j);
            i -= 1;
            break;
          }
          else if (flatSubsequences[j].IsSupersetOf(flatSubsequences[i])) {
            flatSubsequences[i] = flatSubsequences[j];
            deepSubsequences[i] = new List<Subsequence>(deepSubsequences[j]);
            i -= 1;
            break;
          }
        }
      }

      // Remove any duplicate subsequences within each figure.
      for (int i = 0; i < deepSubsequences.Count; ++i)
        deepSubsequences[i] = deepSubsequences[i].Distinct().ToList();

      // Make sure figure constituents are properly ordered.
      for (int i = 0; i < deepSubsequences.Count; ++i) {
        for (int j = 0; j < deepSubsequences[i].Count; ++j)
          deepSubsequences[i][j].OrderBy(s => s.SentenceId).ThenBy(s => s.Left).ToList();
      }

      for (int i = 0; i < deepSubsequences.Count; ++i) {
        deepSubsequences[i] = deepSubsequences[i].OrderBy(s => s[0].SentenceId).ToList();
        for (int j = 0; j < deepSubsequences[i].Count; ++j) {
          var dsij =  deepSubsequences[i][j];
          deepSubsequences[i][j] = new Subsequence(dsij.OrderBy(s => s.Left), dsij.ContainingSentence, dsij.ContainingSubsequence, dsij.WindowId);
          if (demarcation != null)
            deepSubsequences[i][j].Add(new SubsequenceToken(new Token(demarcation, "", 0)));
        }
      }

      var figures = new List<RhetoricalFigure>();
      foreach (var deepSubsequence in deepSubsequences) {
        var d = deepSubsequence.OrderBy(x => x.SentenceId).ThenBy(x => x[0].Left).ToList(); // Sort figure constituents so leftmost in text appears first, etc.
        var figure = new RhetoricalFigure(new Subsequence(d.SelectMany(x => x)), type, d[0].WindowId);
        //figure.Tokens = figure.Tokens.OrderBy(s => s.SentenceId).ThenBy(s => s.Left).ToList(); // This sort could cause problems with the collapsed 'figure'. Stick to the one just above.
        figures.Add(figure);
      }

      return figures;
    }


    /// <summary>
    /// Epizeuxis: Repetition of a word or phrase with no others between.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    public static void FindEpizeuxis(Analyzer a, int? windowSize)
    {
      int ws = windowSize ?? 2; // Use default window size of 2.

      var allSubsequences = new List<List<Subsequence>>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            var phrases = a.Document.Sentences[i + j].Phrases;
            if (phrases.Count > 0)
              window.AddRange(phrases[0].Subsequences);
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          list.Add(new Subsequence(window[j], i));

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current) && comparer.IsRightContiguous(current))
                list.Add(new Subsequence(current, i));
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Epizeuxis, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Ploce: The repetition of word in a short span of text for rhetorical emphasis.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    public static void FindPloce(Analyzer a, int? windowSize)
    {
      int ws = windowSize ?? 2; // Use default window size of 2.

      var allSubsequences = new List<List<Subsequence>>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            var phrases = a.Document.Sentences[i + j].Phrases;
            if (phrases.Count > 0)
              window.AddRange(phrases[0].SubsequencesNoStopWords);
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].Count != 1)
            continue;
          else
            list.Add(new Subsequence(window[j], i));

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current))
                list.Add(new Subsequence(current, i));
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Ploce, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Conduplicatio: The repetition of a word or phrase.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="minLength"></param>
    public static void FindConduplicatio(Analyzer a, int? windowSize, object minLength)
    {
      int ws = windowSize ?? 2; // Use default window size of 2.
      int ml = Convert.ToInt32(minLength ?? 2); // With 'minlength' ≥ 2, this figure might be closer to "epimone."

      var allSubsequences = new List<List<Subsequence>>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            var phrases = a.Document.Sentences[i + j].Phrases;
            if (phrases.Count > 0)
              window.AddRange(phrases[0].Subsequences);
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          //if (window[j].Count < ml)
          if (window[j].Count < ml || window[j].All(t => t.IsStopWord())) // Reject if subsequence contains all stop words.
            continue;
          else
            list.Add(new Subsequence(window[j], i));

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current) && !comparer.IsRightContiguous(current))
                list.Add(new Subsequence(current, i));
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Conduplicatio, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Polysyndeton: "Excessive" repetition of conjunctions between clauses.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="consecutiveStarts"></param>
    public static void FindPolysyndeton(Analyzer a, int? windowSize, object consecutiveStarts)
    {
      int ws = windowSize ?? 1; // Use default window size of 1.
      int cs = Convert.ToInt32(consecutiveStarts ?? 2); // Use default of 2 consecutive sentences for leading polysyndeton.

      var allSubsequences = new List<List<Subsequence>>();

      // Find conjunctions within clauses.
      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            var phrases = a.Document.Sentences[i + j].Phrases;
            if (phrases.Count > 0)
              window.AddRange(phrases[0].Subsequences);
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].Count != 1)
            continue;
          else {
            if (window[j][0].TagEquivalent == "CC")
              list.Add(new Subsequence(window[j], i));
          }

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current))
                list.Add(new Subsequence(current, i));
            }
          }

          if (list.Count > 2) // "Excessive" should mean more than 2.
            allSubsequences.Add(list);
        }
      }

      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Polysyndeton);

      allSubsequences.Clear();

      // Now find conjunctions starting consecutive clauses.
      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < cs; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            var phrases = a.Document.Sentences[i + j].Phrases;
            if (phrases.Count > 0)
              window.AddRange(phrases[0].Subsequences);
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].Count != 1)
            continue;
          else {
            if (window[j][0].TagEquivalent == "CC" && window[j][0].IsStart)
              list.Add(new Subsequence(window[j], i));
          }

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current) && current[0].IsStart)
                list.Add(new Subsequence(current, i));
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }
      }

      // Some figures may be out of order WRT the start of the text; reorder them here.
      //allSubsequences = allSubsequences.OrderBy(s => s[0].SentenceId).ThenBy(s => s[0][0].Left).ToList();

      // Remove duplicate instances and merge those contained in others.
      figures.AddRange(MergeFigures(allSubsequences, RhetoricalFigures.Polysyndeton, multiWindow: true));

      figures = figures.OrderBy(x => x.Tokens.First().SentenceId).ThenBy(x => x.Tokens.First().Left).ToList();

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Anaphora: Repetition of a word or phrase at the beginning of successive clauses.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="minLength"></param>
    public static void FindAnaphora(Analyzer a, int? windowSize, object minLength)
    {
      int ws = windowSize ?? 3; // Use default window size of 3.
      int ml = Convert.ToInt32(minLength ?? 2); // V. Gawryjolek, p. 23

      var allSubsequences = new List<List<Subsequence>>();
      var rejections = new List<Subsequence>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            for (int k = 0; k < a.Document.Sentences[i + j].Clauses.Count; ++k) // Or 'Phrases', but the clauses may be more apt.
              window.AddRange(a.Document.Sentences[i + j].Clauses[k].SubsequencesNoBoundaryDeterminersEtc);
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j][0].IsStart) {
            if (window[j].Count == 1 && window[j].StopWordsStatus.HasFlag(StopWordsOptions.FirstWord)) {
              rejections.Add(new Subsequence(window[j], i));
              continue;
            }
            list.Add(new Subsequence(window[j], i));
          }
          else
            continue;

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current) && current[0].IsStart)
                list.Add(new Subsequence(current, i));
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }
      }

      // Check for false anaphoras and remove them. V. Gawryjolek, p. 59, where a natural epistrophe is incorrectly identified as anaphora.
      rejections = rejections.Distinct().ToList();

      for (int i = 0; i < allSubsequences.Count; ++i) {
        for (int j = 0; j < allSubsequences[i].Count; ++j) {
          var falseAnaphora = false;
          for (int k = 0; k < rejections.Count; ++k) {
            if (allSubsequences[i][j] == rejections[k])
              falseAnaphora = true;
          }
          if (falseAnaphora) {
            allSubsequences.RemoveAt(i);
            i -= 1;
            break;
          }
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Anaphora, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Epistrophe: Repetition of the same word or phrase at the end of successive clauses.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="minLength"></param>
    public static void FindEpistrophe(Analyzer a, int? windowSize, object minLength)
    {
      int ws = windowSize ?? 3; // Use default window size of 3.
      int ml = Convert.ToInt32(minLength ?? 2); // V. Gawryjolek, p. 23

      var allSubsequences = new List<List<Subsequence>>();
      var rejections = new List<Subsequence>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            for (int k = 0; k < a.Document.Sentences[i + j].Clauses.Count; ++k) // Or 'Phrases', but the clauses may be more apt.
              window.AddRange(a.Document.Sentences[i + j].Clauses[k].SubsequencesNoBoundaryDeterminersEtc);
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].Last().IsEnd) {
            if (window[j].Count == 1 && window[j].StopWordsStatus.HasFlag(StopWordsOptions.LastWord)) {
              rejections.Add(new Subsequence(window[j], i));
              continue;
            }
            list.Add(new Subsequence(window[j], i));
          }
          else
            continue;

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current) && current.Last().IsEnd)
                list.Add(new Subsequence(current, i));
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }
      }

      // Check for false epistrophes and remove them. V. Gawryjolek, p. 59, where a natural epistrophe is incorrectly identified as anaphora.
      rejections = rejections.Distinct().ToList();

      for (int i = 0; i < allSubsequences.Count; ++i) {
        for (int j = 0; j < allSubsequences[i].Count; ++j) {
          var falseEpistrophe = false;
          for (int k = 0; k < rejections.Count; ++k) {
            if (allSubsequences[i][j] == rejections[k])
              falseEpistrophe = true;
          }
          if (falseEpistrophe) {
            allSubsequences.RemoveAt(i);
            i -= 1;
            break;
          }
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Epistrophe, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Symploce: Repetition of a word or phrase at the beginning, and of another at the end, of successive clauses; the combination of Anaphora and Epistrophe.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="minLength"></param>
    public static void FindSymploce(Analyzer a, int? windowSize, object minLength)
    {
      int ws = windowSize ?? 3; // Use default window size of 3.
      int ml = Convert.ToInt32(minLength ?? 2); // V. Gawryjolek, p. 23

      var allSubsequences = new List<List<Subsequence>>();
      var rejections = new List<Subsequence>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            for (int k = 0; k < a.Document.Sentences[i + j].Clauses.Count; ++k) { // Or 'Phrases', but the clauses may be more apt.
              var startEndSubsequence = new List<Subsequence>();
              var subsequences = a.Document.Sentences[i + j].Clauses[k].SubsequencesNoBoundaryConjunctions; // Added 29 Mar. 2015.
              //var subsequences = a.Document.Sentences[i + j].Clauses[k].SubsequencesNoBoundaryDeterminersEtc;
              //var subsequences = a.Document.Sentences[i + j].Clauses[k].Subsequences;
              if (subsequences.Count > 0)
                startEndSubsequence.Add(subsequences[0]);
              window.AddRange(startEndSubsequence);
            }
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].Count >= ml && window[j].First().IsStart && window[j].Last().IsEnd) // Some (complete?) redundancy here with the 'IsStart' and 'IsEnd' tests.
            list.Add(new Subsequence(window[j], i));
          else
            continue;

          if (list.Count != 0) {
            var comparer = list.Last();
            for (int k = j + 1; k < window.Count; ++k) {
              var current = new Subsequence(window[k], i);
              var shorter = Math.Min(comparer.Count, current.Count);
              for (int l = 1; l < shorter; ++l) {
                var comparerStart = new Subsequence(comparer.GetRange(0, l), comparer.ContainingSentence, comparer.ContainingSubsequence, comparer.WindowId);
                var currentStart = new Subsequence(current.GetRange(0, l), current.ContainingSentence, current.ContainingSubsequence, current.WindowId);
                for (int m = 1; m < shorter; ++m) {
                  var comparerEnd = new Subsequence(comparer.GetRange(comparer.Count - m, m), comparer.ContainingSentence, comparer.ContainingSubsequence, comparer.WindowId);
                  var currentEnd = new Subsequence(current.GetRange(current.Count - m, m), current.ContainingSentence, current.ContainingSubsequence, current.WindowId);
                  if (comparerStart.Equivalent(currentStart) && comparerEnd.Equivalent(currentEnd)) {
                    var figureList = new List<Subsequence>();
                    comparerEnd.InsertRange(0, comparerStart);
                    currentEnd.InsertRange(0, currentStart);
                    figureList.Add(comparerEnd);
                    figureList.Add(currentEnd);

                    if (figureList.Count > 0)
                      allSubsequences.Add(figureList);
                  }
                }
              }
            }
          }
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Symploce, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Epanalepsis: Repetition at the end of a clause of the word or phrase that began it.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="minLength"></param>
    public static void FindEpanalepsis(Analyzer a, int? windowSize, object minLength)
    {
      int ws = windowSize ?? 3; // Use default window size of 3.
      int ml = Convert.ToInt32(minLength ?? 2); // V. Gawryjolek, p. 23

      var allSubsequences = new List<List<Subsequence>>();

     for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        // No need for the usual search window here; just check every phrase.
        for (int j = 0; j < a.Document.Sentences[i].Clauses.Count; ++j) {
          var window = a.Document.Sentences[i].Clauses[j].SubsequencesNoStartDeterminersEtc;
          //var window = a.Document.Sentences[i].Phrases[j].Subsequences;
          for (int k = 0; k < window.Count; ++k) {
            var list = new List<Subsequence>();
            if (window[k][0].IsStart)
              list.Add(new Subsequence(window[k], j));
            else
              continue;

            if (list.Count != 0) {
              for (int l = k + 1; l < window.Count; ++l) {
                var comparer = list.Last();
                var current = window[l];
                if (comparer.Equivalent(current) && current.Last().IsEnd) {
                  list.Add(new Subsequence(current, j));
                  break;
                }
              }
            }

            if (list.Count > 1)
              allSubsequences.Add(list);
          }
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Epanalepsis, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Anadiplosis: Repetition of the ending word or phrase from the previous clause at the beginning of the next.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="minLength"></param>
    public static void FindAnadiplosis(Analyzer a, int? windowSize, object minLength)
    {
      int ws = windowSize ?? 2; // Use default window size of 2.
      int ml = Convert.ToInt32(minLength ?? 2); // V. Gawryjolek, p. 23

      var allSubsequences = new List<List<Subsequence>>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            var containingSubsequence = new Subsequence();
            if (a.Document.Sentences[i + j].Clauses.Count > 0) {
              var containingSubsequences = a.Document.Sentences[i + j].Clauses[0].SubsequencesNoDeterminersEtc;
              if (containingSubsequences.Count > 0)
                containingSubsequence = a.Document.Sentences[i + j].Clauses[0].SubsequencesNoDeterminersEtc[0]; // No determiners etc. needed here.
              else
                containingSubsequence = a.Document.Sentences[i + j].Clauses[0].Subsequences[0];
            }
            for (int k = 0; k < a.Document.Sentences[i + j].Clauses.Count; ++k) {
              var subsequences = a.Document.Sentences[i + j].Clauses[k].SubsequencesNoStartDeterminersEtc;
              for (int l = 0; l < subsequences.Count; ++l)
                subsequences[l].ContainingSubsequence = containingSubsequence; // To check for contiguity.
              window.AddRange(subsequences);
            }
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].Last().IsEnd)
            list.Add(new Subsequence(window[j], i));
          else
            continue;

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current) && current.First().IsStart) {
                if (comparer.IsRightContiguous(current)) {
                  list.Add(new Subsequence(current, i));
                  break;
                }
              }
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }

        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].First().IsStart)
            list.Add(new Subsequence(window[j], i));
          else
            continue;

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current) && current.Last().IsEnd) {
                if (comparer.IsLeftContiguous(current)) {
                  list.Add(new Subsequence(current, i));
                  break;
                }
              }
            }
          }

          if (list.Count > 1) {
            list = list.OrderBy(s => s.SentenceId).ThenBy(s => s[0].Left).ToList();
            allSubsequences.Add(list);
          }
        }
      }

      // Some figures may be out of order WRT the start of the text; reorder them here.
      //allSubsequences = allSubsequences.OrderBy(s => s[0].SentenceId).ThenBy(s => s[0][0].Left).ToList();

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Anadiplosis, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Antimetabole: Repetition of words in reverse grammatical order.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="minLength"></param>
    public static void FindAntimetabole(Analyzer a, int? windowSize, object minLength)
    {
      int ws = windowSize ?? 1; // Use default window size of 1.
      int ml = Convert.ToInt32(minLength ?? 2);

      var allSubsequences = new List<List<Subsequence>>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            var phrases = a.Document.Sentences[i + j].Phrases;
            if (phrases.Count > 0)
              window.AddRange(phrases[0].SubsequencesKeepNounsVerbsAdjectivesAdverbsTag);
              //window.AddRange(phrases[0].SubsequencesKeepNounsVerbsAdjectivesAdverbsPronounsTagEquivalent);
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          list.Add(new Subsequence(window[j], i));

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = window[k];
              if (comparer.Equivalent(current))
                list.Add(new Subsequence(current, i));
            }
          }

          if (list.Count == 2)
            allSubsequences.Add(list);
        }
      }

      var repetitions = MergeFigures(allSubsequences, RhetoricalFigures.Antimetabole, multiWindow: true, demarcation: null);

      var figures = new List<RhetoricalFigure>();

      for (int i = 0; i < repetitions.Count - 1; ++i) {
        var al = repetitions[i].Tokens.Split();
        for (int j = i + 1; j < repetitions.Count; ++j) {
          if (repetitions[i].WindowId != repetitions[j].WindowId)
            continue;
          var bl = repetitions[j].Tokens.Split();

          if ((al[0].Last().Right <= bl[0].First().Left || al[0].Last().SentenceId < bl[0].First().SentenceId) && 
              (al[1].First().Left >= bl[1].Last().Right || al[1].First().SentenceId > bl[1].Last().SentenceId)) {
            var subsequence = new Subsequence();
            subsequence.AddRange(al[0]);
            bl[0].Add(new SubsequenceToken(new Token(FigureComponentsSeparator, "", 0)));
            subsequence.AddRange(bl[0]);
            subsequence.AddRange(bl[1]);
            al[1].Add(new SubsequenceToken(new Token(FigureComponentsSeparator, "", 0)));
            subsequence.AddRange(al[1]);

            figures.Add(new RhetoricalFigure(subsequence, RhetoricalFigures.Antimetabole, repetitions[i].WindowId));
          }
        }
      }

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Polyptoton: Repetition of a word in a different form; having cognate words in close proximity.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    public static void FindPolyptoton(Analyzer a, int? windowSize)
    {
      int ws = windowSize ?? 3; // Use default window size of 3.

      var allSubsequences = new List<List<Subsequence>>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            var phrases = a.Document.Sentences[i + j].Phrases;
            if (phrases.Count > 0)
              window.AddRange(phrases[0].SubsequencesNoStopWords);
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].Count != 1)
            continue;
          else
            list.Add(new Subsequence(window[j], i));

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              if (window[k].Count != 1 || list.Last().Equivalent(window[k]))
                continue;
              var comparer = list.Last().Last().DerivationalForms;
              var current = window[k].Last().DerivationalForms;
              if (comparer.Intersect(current).Any())
                list.Add(new Subsequence(window[k], i));
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Polyptoton, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Isocolon: Repetition of grammatical structure in nearby phrases or clauses of approximately equal length.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="similarityThresholdObject"></param>
    public static void FindIsocolon(Analyzer a, int? windowSize, object similarityThresholdObject)
    {
      int ws = windowSize ?? 3; // Use default window size of 3.
      int similarityThreshold = Convert.ToInt32(similarityThresholdObject ?? 0); // Was 1, but that's perhaps too greedy.

      int minPhraseLength = 2;

      var allSubsequences = new List<List<Subsequence>>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) { 
            for (int k = 0; k < a.Document.Sentences[i + j].Phrases.Count; ++k) {
              var first = new List<Subsequence>(a.Document.Sentences[i + j].Phrases[k].Subsequences);
              if (first.Count == 0) // Necessary when the parser encounters certain non-standard punctuation -- make sure text is clean to avoid hitting it!
                continue;
              first.RemoveRange(1, first.Count - 1);
              window.AddRange(first);
            }
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].Count < minPhraseLength)
            continue;
          list.Add(new Subsequence(window[j], i));

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last(); // Was 'list.First()'.
              var current = window[k];
              if (current.Count < minPhraseLength)
                continue;
              if (comparer - current <= similarityThreshold) {
                if (!comparer.IsSupersetOf(current) &&
                    comparer[0].TagEquivalent == current[0].TagEquivalent && comparer.Last().TagEquivalent == current.Last().TagEquivalent) // Changed 28 Mar. 2015.
                    //comparer[0].Tag == current[0].Tag && comparer.Last().TagEquivalent == current.Last().TagEquivalent)
                    //comparer[0].Tag == current[0].Tag && comparer.Last().Tag == current.Last().Tag)
                  list.Add(new Subsequence(current, i));
              }
            }
          }

          // Remove any subsequences that are subsets of other subsequences.
          for (int k = 0; k < list.Count - 1; ++k) {
            for (int l = k + 1; l < list.Count; ++l) {
              if (list[k].IsSupersetOf(list[l])) {
                list.RemoveAt(l);
                l -= 1;
                continue;
              }
              else if (list[l].IsSupersetOf(list[k])) {
                list[k] = list[l];
                list.RemoveAt(l);
                k = 0;
                break;
              }
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Isocolon, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    /// <summary>
    /// Chiasmus: Repetition of grammatical structures in reverse order.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    /// <param name="minLength"></param>
    public static void FindChiasmus(Analyzer a, int? windowSize, object minLength)
    {
      int ws = windowSize ?? 3; // Use default window size of 3.
      int ml = Convert.ToInt32(minLength ?? 3);

      var allSubsequences = new List<List<Subsequence>>();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var window = new List<Subsequence>(); // Search window
        var reverseWindow = new List<Subsequence>();
        for (int j = 0; j < ws; ++j) {
          if (i + j < a.Document.Sentences.Count) {
            var pptp = a.Document.Sentences[i + j].GetPrePreTerminalPhrases();
            //pptp.AddRange(a.Document.Sentences[i + j].Phrases); pptp = pptp.Distinct().ToList(); // Test code.
            var pptpSubsequences = new List<Subsequence>();
            for (int k = 0; k < pptp.Count; ++k)
              pptpSubsequences.Add(pptp[k].Subsequences[0]);
            var pptpContiguousSubsequences = pptpSubsequences.ContiguousSubsequences().ToList();

            foreach (var s in pptpContiguousSubsequences) {
              if (s.Count > 1) { // Because reversal on a single element returns the same element.
                window.Add(new Subsequence(s.SelectMany(x => x), a.Document.Sentences[i + j], s[0], s[0].WindowId));
                reverseWindow.Add(new Subsequence(s.Reverse().SelectMany(x => x), a.Document.Sentences[i + j], s[0], s[0].WindowId));
              }
            }
          }
        }

        // Search.
        for (int j = 0; j < window.Count; ++j) {
          var list = new List<Subsequence>();
          if (window[j].Count < ml)
            continue;
          list.Add(new Subsequence(window[j], i));

          if (list.Count != 0) {
            for (int k = j + 1; k < window.Count; ++k) {
              var comparer = list.Last();
              var current = reverseWindow[k];
              if (current.Count < ml
                || comparer.First().TagEquivalent == "CC" || comparer.Last().TagEquivalent == "CC")
                continue;
              if (comparer.EqualsInTagEquivalent(current) && !comparer.Intersect(current).Any()) {
                list.Add(new Subsequence(current, i));
                break;
              }
            }
          }

          if (list.Count > 1)
            allSubsequences.Add(list);
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Chiasmus, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    public delegate int GetDependencyIndexDelegate(TreeGraphNode td);

    public class OxymoronData
    {
      public Token[] Pair { get; protected set;}
      public string W1 {
        get { return Pair[0].Word.ToLower(); }
      }

      protected bool _greedy = false;
      public bool Greedy
      {
        get { return _greedy; }
        set { _greedy = value; }
      }

      protected bool _debug = false;
      public bool Debug {
        get { return _debug; }
        set { _debug = value; }
      }

      public string W2
      {
        get { return Pair[1].Word.ToLower(); }
      }
 
      private readonly Lazy<List<string>> _derivedFormsW2;

      public IntClass Overlap { get; set; }

      public class IntClass // To create a reference variable based on type 'int'.
      {
        public int Value { get; set; }

        public IntClass(int value)
        {
          Value = value;
        }
      }

      public OxymoronData(Token[] pair)
      {
        Pair = pair;

        _derivedFormsW2 = new Lazy<List<string>>(GetDerivedFormsW2);
      }

      public OxymoronData(Token[] pair, IntClass overlap)
        : this(pair)
      {
        Overlap = overlap;
      }

      public OxymoronData(Token[] pair, IntClass overlap, bool greedy)
        : this(pair, overlap)
      {
        Greedy = greedy;
      }

      public OxymoronData(Token[] pair, IntClass overlap, bool greedy, bool debug)
        : this(pair, overlap, greedy)
      {
        Debug = debug;
      }

      public List<string> GetDerivedFormsW2()
      {
        if (this.Greedy)
          return Token.FindSynonyms(Token.FindDerivationalForms(W2, Pair[1].Stem, Analyzer.SimilarityPrefixes, Analyzer.MostCommonSimilaritySuffixes, false, removeCandidate: false));
        else
          return Token.FindDerivationalForms(W2, Pair[1].Stem, Analyzer.SimilarityPrefixes, Analyzer.MostCommonSimilaritySuffixes, false, removeCandidate: false);
      }
    }

    /// <summary>
    /// Oxymoron: A terse paradox; the yoking of two contradictory terms.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="windowSize"></param>
    public static void FindOxymoron(Analyzer a, int? windowSize, object greedy) // Add WordNet search paths to this as the 'object' parameter?
    {
      int ws = windowSize ?? 1; // Not used. The window size is one sentence.
      bool greedySearch = (bool?)greedy ?? false;

      GetDependencyIndexDelegate GetDependencyIndex = delegate(TreeGraphNode t)
      {
        return Convert.ToInt32(Regex.Match(t.toString(), "^.*?-(\\d+)\\'*$").Result("$1")) - 1;
      };

      Action<Miscellaneous.TreeNode<Analyzer.WordNetRelation>, object> WordNetRelationVisitor =
        (Miscellaneous.TreeNode<Analyzer.WordNetRelation> n, object o) =>
      {
        if (n.IsRoot())
          return;

        var oxymoronData = (OxymoronData)o;

        if (oxymoronData.Overlap.Value != 0)
          return;

        var w1 = oxymoronData.W1;
        var derivedFormsW2 = oxymoronData.GetDerivedFormsW2();

        bool checkedAntonyms = false;
        var currentNode = n;
        while (!currentNode.Parent.IsRoot()) {
          currentNode = currentNode.Parent;
          if (currentNode.Value.Relation == WordNetEngine.SynSetRelation.Antonym) {
            checkedAntonyms = true;
            break;
          }
        }

        var p = n.Parent;

        var candidates = new List<string> { w1 };
        if (!p.IsRoot())
          candidates = p.Value.Words;

        var relation = n.Value.Relation;

        switch(relation) {
          case WordNetEngine.SynSetRelation.SimilarTo:
            n.Value.Words = Token.FindSynonyms(candidates);
            break;

          case WordNetEngine.SynSetRelation.Antonym:
            n.Value.Words = Token.FindAntonyms(candidates);
            if (!checkedAntonyms)
              checkedAntonyms = true;
            break;

          case WordNetEngine.SynSetRelation.DerivationallyRelated:
            n.Value.Words = Token.FindDerivationalForms(candidates, Analyzer.SimilarityPrefixes, Analyzer.MostCommonSimilaritySuffixes, useAllForms: greedySearch ? true : false);
            if (checkedAntonyms) {
              var negations = new List<string>(Analyzer.NegationPrefixes.Select(x => (string)(x.Clone()) + w1));

              n.Value.Words.AddRange(Token.FindDerivationalForms(negations, null, null, useAllForms: greedySearch ? true : false));
            }
            break;
        }

        if (!checkedAntonyms)
          n.Value.Words.AddRange(candidates);

        n.Value.Words = n.Value.Words.Distinct().ToList(); // Remove duplicates.

        if (oxymoronData.Debug) {
          Console.WriteLine("===================================================");
          Console.WriteLine("Relation: " + relation.ToString());
          //Console.WriteLine("Parent relation: " + p.Value.Relation.ToString());
          Console.WriteLine("Child count: " + n.Children.Count());
          Console.WriteLine("Node candidates:");
          if (n.IsRoot() || n.Value.Words.Count == 0) Console.WriteLine("  None");
          else {
            foreach (var w in n.Value.Words)
              Console.WriteLine("  " + w.ToString());
          }
          if (n.IsLeaf()) Console.WriteLine("LEAF NODE");
          Console.WriteLine("===================================================");
        }

        if (checkedAntonyms)
          oxymoronData.Overlap.Value = n.Value.Words.Intersect(derivedFormsW2).Count();
      };

      Action<Miscellaneous.TreeNode<Analyzer.WordNetRelation>, object> WordNetRelationNullVisitor =
        (Miscellaneous.TreeNode<Analyzer.WordNetRelation> n, object o) =>
      {
        //Console.WriteLine(n.Value.Relation.ToString());
        n.Value.Words = null;
      };

      string dependencySymbols = @"^(amod|advmod|acomp|dobj|nsubj|prep)$";

      var allSubsequences = new List<List<Subsequence>>();

      TreebankLanguagePack tlp = new PennTreebankLanguagePack();
      GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();

      for (int i = 0; i < a.Document.Sentences.Count; ++i) {
        var sentence = a.Document.Sentences[i];
        var subsequenceTokens = new List<SubsequenceToken>();
        foreach (var token in sentence.Tokens)
          subsequenceTokens.Add(new SubsequenceToken(token, sentence));
        var phrases = sentence.Phrases;
        if (phrases.Count > 0) {
          var subsequence = new Subsequence(subsequenceTokens, sentence, phrases[0].Subsequences[0].ContainingSubsequence, i);

          var tree = sentence.Tree;
          GrammaticalStructure gs = gsf.newGrammaticalStructure(tree);
          java.util.Collection tdc = gs.typedDependenciesCollapsed();

          var candidates = new List<Subsequence>();
          for (java.util.Iterator j = tdc.iterator(); j.hasNext(); ) {
            var td = (TypedDependency)j.next();
            var relation = td.reln().getShortName();
            if (Regex.IsMatch(relation, dependencySymbols)) {
              var governorIndex = GetDependencyIndex(td.gov());
              var dependentIndex = GetDependencyIndex(td.dep());

              var index = Math.Min(governorIndex, dependentIndex);
              var count = Math.Abs(dependentIndex - governorIndex) + 1;
              var ss = relation == "prep" ? subsequence.GetRange(index, count) : subsequence.Where((n, k) => k == governorIndex | k == dependentIndex).ToList();

              // Remove any leftover punctuation from the candidate subsequences.
              ss.RemoveAll(n => Regex.IsMatch(n.Tag, Analyzer.PunctuationPatterns));

              candidates.Add(new Subsequence(ss, sentence, subsequence.ContainingSubsequence, i));
            }
          }

          // Determine whether the candidate pairs are oxymorons.
          for (int k = 0; k < candidates.Count; ++k) {
            var list = new List<Subsequence>();

            Token[] pair = { candidates[k][0], candidates[k][candidates[k].Count - 1] };

            // Clear (i.e. null) all the word lists in the WordNet search-path tree.
            a.WordNetSearchPath.Traverse(WordNetRelationNullVisitor);

            var overlap = new OxymoronData.IntClass(0);
            a.WordNetSearchPath.Traverse(WordNetRelationVisitor, new OxymoronData(pair, overlap, greedy: greedySearch, debug: false));
            if (overlap.Value == 0) {
              a.WordNetSearchPath.Traverse(WordNetRelationNullVisitor);
              a.WordNetSearchPath.Traverse(WordNetRelationVisitor, new OxymoronData(pair.Reverse().ToArray(), overlap, greedy: greedySearch, debug: false));
            }

            if (overlap.Value != 0) {
              list.Add(candidates[k]);
              allSubsequences.Add(list);
            }
          }
        }
      }

      // Remove duplicate instances and merge those contained in others.
      var figures = MergeFigures(allSubsequences, RhetoricalFigures.Oxymoron, multiWindow: true);

      a.Figures.AddRange(figures);
    }


    public string Text
    {
      get
      {
        string s = string.Empty;
        s += "W" + WindowId + ": [" + this.Type + "] ";
        foreach (var token in Tokens)
          s += token.Word + " ";

        return s.Trim();
      }
    }

    // Operators, indexers, and events:
  }
}
