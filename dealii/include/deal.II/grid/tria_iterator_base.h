// ---------------------------------------------------------------------
//
// Copyright (C) 1999 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_tria_iterator_base_h
#define dealii_tria_iterator_base_h


#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

/**
 * Namespace in which an enumeration is declared that denotes the states in
 * which an iterator can be in.
 *
 * @ingroup Iterators
 */
namespace IteratorState
{
  /**
   * The three states an iterator can be in: valid, past-the-end and invalid.
   */
  enum IteratorStates
  {
    /// Iterator points to a valid object
    valid,
    /// Iterator reached end of container
    past_the_end,
    /// Iterator is invalid, probably due to an error
    invalid
  };
} // namespace IteratorState



DEAL_II_NAMESPACE_CLOSE

#endif
