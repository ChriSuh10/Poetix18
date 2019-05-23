using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

using IKVM.Runtime;

using edu.stanford.nlp.ling;
using edu.stanford.nlp.trees;

using Rhetorica;

namespace ExtensionMethods
{
  public static class TreeExtensions
  {
    public static Phrase GetTokens(this Tree tree, Tree root = null, Rhetorica.Sentence sentence = null, string ignore = "", string punctuation = null, AnalyzerOptions options = AnalyzerOptions.None)
    {
      var tokens = new Phrase(sentence: sentence);
      java.util.List leaves = tree.getLeaves();

      for (java.util.Iterator i = leaves.iterator(); i.hasNext(); ) {
        Tree leaf = (Tree)i.next();
        string token = leaf.value().Trim();

        Tree preterminal = leaf.parent(tree);
        if (preterminal == null)
          continue;
        string tag = preterminal.value().Trim();

        bool ignoreMeansInclude = options.HasFlag(AnalyzerOptions.IgnoreMeansInclude);
        if (ignore != string.Empty) {
          bool isMatch = Regex.IsMatch(token, ignore);
          if (ignoreMeansInclude) {
            if (!isMatch) continue;
          }
          else {
            if (isMatch) continue;
          }
        }

        bool omitPunctuation = options.HasFlag(AnalyzerOptions.OmitPunctuationTokens);
        if (omitPunctuation) {
          // Leave out certain types of punctuation:
          bool isPunctuation = Regex.IsMatch(tag, punctuation ?? Analyzer.PunctuationPatterns)
            || Regex.IsMatch(token, punctuation ?? Analyzer.PunctuationPatterns);
          if (isPunctuation) {
            tokens.IsPunctuationOmitted = true;
            continue;
          }

          // But also remove any straggler punctuation missed within a token...? Maybe not. Use RegExp 'FloatingPunctuationPatterns' if so.
        }

        root = root ?? tree;
        int depth = root.depth() - root.depth(preterminal);

        var characterEdges = new CharacterEdges(root.leftCharEdge(leaf), root.rightCharEdge(leaf));
        tokens.Add(new Token(token, tag, depth, characterEdges));
      }

      return tokens;
    }

    public static List<Phrase> GetPhrases(this Tree root, Rhetorica.Sentence sentence = null, string ignore = "", string punctuation = null, AnalyzerOptions options = AnalyzerOptions.None)
    {
      var phrases = new List<Phrase>();

      for (java.util.Iterator i = root.iterator(); i.hasNext(); ) {
        Tree tree = (Tree)i.next();
        if (tree.isPhrasal()) {
          java.util.List children = tree.getChildrenAsList();
          if (children.size() == 1 && ((Tree)children.get(0)).isPhrasal())
            continue;

          var current = new Phrase(tree.GetTokens(root, sentence, ignore, punctuation, options));
          // If current node matches previous node but for punctuation omission, replace previous with current:
          bool omitFalseDuplicatePhrases = options.HasFlag(AnalyzerOptions.OmitFalseDuplicatePhrases);
          if (omitFalseDuplicatePhrases) {
            if (phrases.Count > 0) {
              Phrase previous = phrases.Last();
              if (previous.EqualExceptPunctuationOmission(current)) {
                phrases[phrases.Count - 1] = current;
                continue;
              }
            }
          }

          if (current.Count == 0)
            continue;

          phrases.Add(current);
        }
      }

      return phrases;
    }

    public static List<Phrase> GetPrePreTerminalPhrases(this Tree root, Rhetorica.Sentence sentence = null, string ignore = "", string punctuation = null, AnalyzerOptions options = AnalyzerOptions.None)
    {
      var phrases = new List<Phrase>();

      for (java.util.Iterator i = root.iterator(); i.hasNext(); ) {
        Tree tree = (Tree)i.next();
        if (tree.isPreTerminal() || tree.isPrePreTerminal()) {
          if (tree.isPreTerminal() && tree.parent(root) != null) {
            if (tree.parent(root).isPrePreTerminal())
              continue;
          }

          var current = new Phrase(tree.GetTokens(root, sentence, ignore, punctuation, options));
          // If current node matches previous node but for punctuation omission, replace previous with current:
          bool omitFalseDuplicatePhrases = options.HasFlag(AnalyzerOptions.OmitFalseDuplicatePhrases);
          if (omitFalseDuplicatePhrases) {
            if (phrases.Count > 0) {
              Phrase previous = phrases.Last();
              if (previous.EqualExceptPunctuationOmission(current)) {
                phrases[phrases.Count - 1] = current;
                continue;
              }
            }
          }

          if (current.Count == 0)
            continue;

          phrases.Add(current);
        }
      }

      // If "phrase" is a single token which is a preposition (IN) or infinitival to (TO), then join it to the subsequent phrase.
      for (int i = 0; i < phrases.Count; ++i) {
        if (phrases[i].Count == 1 && Regex.IsMatch(phrases[i][0].TagEquivalent, @"^(IN|TO)$", RegexOptions.IgnoreCase) && i != phrases.Count - 1) {
          phrases[i + 1].Tokens.InsertRange(0, phrases[i].Tokens);
          phrases.RemoveAt(i);
          i =- 1;
        }
      }

      return phrases;
    }

    public static List<Phrase> GetClauses(this Tree root, Rhetorica.Sentence sentence = null, string ignore = "", string punctuation = null, AnalyzerOptions options = AnalyzerOptions.None)
    {
      var phrases = new List<Phrase>();

      for (java.util.Iterator i = root.iterator(); i.hasNext(); ) {
        Tree tree = (Tree)i.next();

        var treeLabel = tree.label().value();
        var clauseRe = @"^(S|SBAR|SBARQ|SINV|SQ|FRAG)$";
        bool isClausal = Regex.IsMatch(treeLabel, clauseRe, RegexOptions.IgnoreCase);

        if (isClausal) {
          var current = new Phrase(tree.GetTokens(root, sentence, ignore, punctuation, options));
          // If current node matches previous node but for punctuation omission, replace previous with current:
          bool omitFalseDuplicatePhrases = options.HasFlag(AnalyzerOptions.OmitFalseDuplicatePhrases);
          if (omitFalseDuplicatePhrases) {
            if (phrases.Count > 0) {
              Phrase previous = phrases.Last();
              if (previous.EqualExceptPunctuationOmission(current)) {
                phrases[phrases.Count - 1] = current;
                continue;
              }
            }
          }

          if (current.Count == 0)
            continue;

          phrases.Add(current);
        }
      }

      if (phrases.Count == 0) { // Since 'root' has been identified as a sentence, it should have at least one clause associated with it.
        var pseudoClauses = root.GetPhrases(sentence, ignore, punctuation, options);
        if (pseudoClauses.Count > 0)
          phrases.Add(pseudoClauses[0]);
      }

      return phrases;
    }
  }
}
